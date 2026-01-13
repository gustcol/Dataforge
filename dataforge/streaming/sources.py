"""
DataForge Stream Sources

Streaming data sources for Spark Structured Streaming.

Supported Sources:
    - Kafka: Apache Kafka topics
    - File: CSV, JSON, Parquet, ORC files
    - Delta: Delta Lake tables with change feed
    - Rate: Synthetic rate source for testing

Best Practices:
    1. Use Auto Loader for file sources (better scalability)
    2. Configure appropriate starting offsets for Kafka
    3. Use Delta CDF for incremental processing
    4. Set max files per trigger for file sources
    5. Consider schema inference vs explicit schema

Example:
    >>> # Kafka source
    >>> kafka_source = KafkaSource(spark, "kafka:9092", ["events"])
    >>> stream = kafka_source.read_stream()
    >>>
    >>> # File source with Auto Loader
    >>> file_source = FileSource(spark, "s3://bucket/data", "json")
    >>> stream = file_source.read_stream(use_auto_loader=True)
    >>>
    >>> # Delta source with change feed
    >>> delta_source = DeltaSource(spark, "catalog.schema.events")
    >>> stream = delta_source.read_stream(read_change_feed=True)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING
import logging

if TYPE_CHECKING:
    from pyspark.sql import SparkSession, DataFrame
    from pyspark.sql.types import StructType

logger = logging.getLogger(__name__)


@dataclass
class SourceConfig:
    """Configuration for stream sources.

    Attributes:
        max_files_per_trigger: Max files to process per trigger
        max_bytes_per_trigger: Max bytes to process per trigger
        schema: Optional schema (None for inference)
        options: Additional source-specific options
    """
    max_files_per_trigger: Optional[int] = None
    max_bytes_per_trigger: Optional[str] = None
    schema: Optional["StructType"] = None
    options: Dict[str, str] = field(default_factory=dict)


class StreamSource(ABC):
    """
    Abstract base class for stream sources.

    All stream sources must implement read_stream() method.
    """

    def __init__(self, spark: "SparkSession") -> None:
        """
        Initialize stream source.

        Args:
            spark: SparkSession
        """
        self.spark = spark

    @abstractmethod
    def read_stream(self) -> "DataFrame":
        """
        Read streaming DataFrame from source.

        Returns:
            Streaming DataFrame
        """
        pass


class KafkaSource(StreamSource):
    """
    Kafka stream source.

    Reads streaming data from Apache Kafka topics.

    Example:
        >>> source = KafkaSource(
        ...     spark,
        ...     bootstrap_servers="kafka:9092",
        ...     topics=["events", "logs"]
        ... )
        >>> stream = source.read_stream()
        >>>
        >>> # Parse JSON value
        >>> parsed = stream.selectExpr(
        ...     "CAST(key AS STRING)",
        ...     "CAST(value AS STRING)",
        ...     "timestamp"
        ... )
    """

    def __init__(
        self,
        spark: "SparkSession",
        bootstrap_servers: str,
        topics: List[str],
        starting_offsets: str = "latest",
        consumer_group: Optional[str] = None,
        fail_on_data_loss: bool = True,
        include_headers: bool = False
    ) -> None:
        """
        Initialize Kafka source.

        Args:
            spark: SparkSession
            bootstrap_servers: Kafka bootstrap servers
            topics: List of topic names
            starting_offsets: "earliest", "latest", or JSON offset spec
            consumer_group: Optional consumer group ID
            fail_on_data_loss: Fail if data loss detected
            include_headers: Include Kafka headers in output
        """
        super().__init__(spark)
        self.bootstrap_servers = bootstrap_servers
        self.topics = topics
        self.starting_offsets = starting_offsets
        self.consumer_group = consumer_group
        self.fail_on_data_loss = fail_on_data_loss
        self.include_headers = include_headers

    def read_stream(self) -> "DataFrame":
        """Read streaming DataFrame from Kafka."""
        options = {
            "kafka.bootstrap.servers": self.bootstrap_servers,
            "subscribe": ",".join(self.topics),
            "startingOffsets": self.starting_offsets,
            "failOnDataLoss": str(self.fail_on_data_loss).lower(),
            "includeHeaders": str(self.include_headers).lower(),
        }

        if self.consumer_group:
            options["kafka.group.id"] = self.consumer_group

        reader = self.spark.readStream.format("kafka")

        for key, value in options.items():
            reader = reader.option(key, value)

        logger.info(f"Reading stream from Kafka topics: {self.topics}")
        return reader.load()

    def read_batch(
        self,
        starting_offsets: Optional[str] = None,
        ending_offsets: str = "latest"
    ) -> "DataFrame":
        """
        Read batch DataFrame from Kafka.

        Useful for backfilling or testing.

        Args:
            starting_offsets: Starting offsets (default: use instance setting)
            ending_offsets: Ending offsets

        Returns:
            Batch DataFrame
        """
        options = {
            "kafka.bootstrap.servers": self.bootstrap_servers,
            "subscribe": ",".join(self.topics),
            "startingOffsets": starting_offsets or self.starting_offsets,
            "endingOffsets": ending_offsets,
        }

        reader = self.spark.read.format("kafka")

        for key, value in options.items():
            reader = reader.option(key, value)

        return reader.load()


class FileSource(StreamSource):
    """
    File stream source.

    Reads streaming data from files (CSV, JSON, Parquet, etc.).

    Example:
        >>> # JSON files
        >>> source = FileSource(spark, "s3://bucket/events", "json")
        >>> stream = source.read_stream()
        >>>
        >>> # With Auto Loader (Databricks)
        >>> source = FileSource(spark, "s3://bucket/events", "json")
        >>> stream = source.read_stream(use_auto_loader=True)
        >>>
        >>> # With explicit schema
        >>> source = FileSource(spark, "s3://bucket/events", "parquet")
        >>> source.schema = my_schema
        >>> stream = source.read_stream()
    """

    def __init__(
        self,
        spark: "SparkSession",
        path: str,
        format: str = "parquet",
        schema: Optional["StructType"] = None,
        max_files_per_trigger: Optional[int] = None
    ) -> None:
        """
        Initialize file source.

        Args:
            spark: SparkSession
            path: Source path (local, S3, ADLS, etc.)
            format: File format (json, csv, parquet, orc)
            schema: Optional schema (None for inference)
            max_files_per_trigger: Max files per micro-batch
        """
        super().__init__(spark)
        self.path = path
        self.format = format
        self.schema = schema
        self.max_files_per_trigger = max_files_per_trigger

    def read_stream(
        self,
        use_auto_loader: bool = False,
        options: Optional[Dict[str, str]] = None
    ) -> "DataFrame":
        """
        Read streaming DataFrame from files.

        Args:
            use_auto_loader: Use Databricks Auto Loader
            options: Additional options

        Returns:
            Streaming DataFrame
        """
        if use_auto_loader:
            return self._read_with_auto_loader(options)

        reader = self.spark.readStream.format(self.format)

        if self.schema:
            reader = reader.schema(self.schema)

        if self.max_files_per_trigger:
            reader = reader.option("maxFilesPerTrigger", self.max_files_per_trigger)

        if options:
            for key, value in options.items():
                reader = reader.option(key, value)

        logger.info(f"Reading stream from {self.path} ({self.format})")
        return reader.load(self.path)

    def _read_with_auto_loader(
        self,
        options: Optional[Dict[str, str]] = None
    ) -> "DataFrame":
        """Read using Databricks Auto Loader."""
        reader = (
            self.spark.readStream
            .format("cloudFiles")
            .option("cloudFiles.format", self.format)
        )

        if self.schema:
            reader = reader.schema(self.schema)
        else:
            reader = reader.option("cloudFiles.inferColumnTypes", "true")

        if self.max_files_per_trigger:
            reader = reader.option("cloudFiles.maxFilesPerTrigger", self.max_files_per_trigger)

        if options:
            for key, value in options.items():
                reader = reader.option(key, value)

        logger.info(f"Reading stream with Auto Loader from {self.path}")
        return reader.load(self.path)


class DeltaSource(StreamSource):
    """
    Delta Lake stream source.

    Reads streaming data from Delta Lake tables.

    Example:
        >>> # Standard streaming read
        >>> source = DeltaSource(spark, "catalog.schema.events")
        >>> stream = source.read_stream()
        >>>
        >>> # With Change Data Feed
        >>> source = DeltaSource(spark, "catalog.schema.events")
        >>> stream = source.read_stream(read_change_feed=True)
        >>>
        >>> # Starting from specific version
        >>> source = DeltaSource(spark, "catalog.schema.events")
        >>> stream = source.read_stream(starting_version=100)
    """

    def __init__(
        self,
        spark: "SparkSession",
        table_name: str,
        max_files_per_trigger: Optional[int] = None,
        max_bytes_per_trigger: Optional[str] = None
    ) -> None:
        """
        Initialize Delta source.

        Args:
            spark: SparkSession
            table_name: Delta table name (catalog.schema.table)
            max_files_per_trigger: Max files per micro-batch
            max_bytes_per_trigger: Max bytes per micro-batch (e.g., "10g")
        """
        super().__init__(spark)
        self.table_name = table_name
        self.max_files_per_trigger = max_files_per_trigger
        self.max_bytes_per_trigger = max_bytes_per_trigger

    def read_stream(
        self,
        read_change_feed: bool = False,
        starting_version: Optional[int] = None,
        starting_timestamp: Optional[str] = None,
        ignore_deletes: bool = False,
        ignore_changes: bool = False
    ) -> "DataFrame":
        """
        Read streaming DataFrame from Delta table.

        Args:
            read_change_feed: Enable Change Data Feed
            starting_version: Start from specific version
            starting_timestamp: Start from specific timestamp
            ignore_deletes: Ignore delete operations
            ignore_changes: Ignore update operations

        Returns:
            Streaming DataFrame
        """
        reader = self.spark.readStream.format("delta")

        if self.max_files_per_trigger:
            reader = reader.option("maxFilesPerTrigger", self.max_files_per_trigger)

        if self.max_bytes_per_trigger:
            reader = reader.option("maxBytesPerTrigger", self.max_bytes_per_trigger)

        if read_change_feed:
            reader = reader.option("readChangeFeed", "true")

        if starting_version is not None:
            reader = reader.option("startingVersion", starting_version)

        if starting_timestamp:
            reader = reader.option("startingTimestamp", starting_timestamp)

        if ignore_deletes:
            reader = reader.option("ignoreDeletes", "true")

        if ignore_changes:
            reader = reader.option("ignoreChanges", "true")

        logger.info(f"Reading stream from Delta table: {self.table_name}")
        return reader.table(self.table_name)


class RateSource(StreamSource):
    """
    Rate stream source for testing.

    Generates synthetic streaming data at a specified rate.

    Example:
        >>> source = RateSource(spark, rows_per_second=100)
        >>> stream = source.read_stream()
        >>>
        >>> # Generate for 1 minute of simulated time
        >>> source = RateSource(spark, rows_per_second=100, num_partitions=4)
        >>> stream = source.read_stream()
    """

    def __init__(
        self,
        spark: "SparkSession",
        rows_per_second: int = 1,
        num_partitions: Optional[int] = None,
        ramp_up_time: Optional[str] = None
    ) -> None:
        """
        Initialize rate source.

        Args:
            spark: SparkSession
            rows_per_second: Rows generated per second
            num_partitions: Number of output partitions
            ramp_up_time: Time to ramp up to full rate (e.g., "5s")
        """
        super().__init__(spark)
        self.rows_per_second = rows_per_second
        self.num_partitions = num_partitions
        self.ramp_up_time = ramp_up_time

    def read_stream(self) -> "DataFrame":
        """
        Read synthetic streaming DataFrame.

        Returns:
            Streaming DataFrame with columns: timestamp, value
        """
        reader = (
            self.spark.readStream
            .format("rate")
            .option("rowsPerSecond", self.rows_per_second)
        )

        if self.num_partitions:
            reader = reader.option("numPartitions", self.num_partitions)

        if self.ramp_up_time:
            reader = reader.option("rampUpTime", self.ramp_up_time)

        logger.info(f"Reading rate stream: {self.rows_per_second} rows/sec")
        return reader.load()


class SocketSource(StreamSource):
    """
    Socket stream source for testing.

    Reads streaming data from a TCP socket.

    Warning: For testing only. Not for production use.

    Example:
        >>> source = SocketSource(spark, host="localhost", port=9999)
        >>> stream = source.read_stream()
    """

    def __init__(
        self,
        spark: "SparkSession",
        host: str = "localhost",
        port: int = 9999
    ) -> None:
        """
        Initialize socket source.

        Args:
            spark: SparkSession
            host: Socket host
            port: Socket port
        """
        super().__init__(spark)
        self.host = host
        self.port = port

    def read_stream(self) -> "DataFrame":
        """
        Read streaming DataFrame from socket.

        Returns:
            Streaming DataFrame with column: value
        """
        logger.warning("Socket source is for testing only. Not for production use.")

        return (
            self.spark.readStream
            .format("socket")
            .option("host", self.host)
            .option("port", self.port)
            .load()
        )
