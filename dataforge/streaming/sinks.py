"""
DataForge Stream Sinks

Streaming data sinks for Spark Structured Streaming.

Supported Sinks:
    - Kafka: Write to Apache Kafka topics
    - Delta: Write to Delta Lake tables (exactly-once)
    - File: Write to files (JSON, Parquet, etc.)
    - Console: Display output (debugging)
    - ForeachBatch: Custom batch processing

Best Practices:
    1. Use Delta Lake for exactly-once semantics
    2. Always configure checkpoint locations
    3. Choose appropriate output modes
    4. Monitor trigger latency
    5. Use foreachBatch for complex sinks

Example:
    >>> # Delta sink with merge
    >>> sink = DeltaSink(spark, "catalog.schema.events")
    >>> query = sink.write_stream(
    ...     stream,
    ...     checkpoint_path="/checkpoints/events",
    ...     output_mode="append"
    ... )
    >>>
    >>> # Kafka sink
    >>> sink = KafkaSink(spark, "kafka:9092", "output-topic")
    >>> query = sink.write_stream(stream, checkpoint_path="/checkpoints/kafka")
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, TYPE_CHECKING
import logging

if TYPE_CHECKING:
    from pyspark.sql import SparkSession, DataFrame
    from pyspark.sql.streaming import StreamingQuery

logger = logging.getLogger(__name__)


class StreamSink(ABC):
    """
    Abstract base class for stream sinks.

    All stream sinks must implement write_stream() method.
    """

    def __init__(self, spark: "SparkSession") -> None:
        """
        Initialize stream sink.

        Args:
            spark: SparkSession
        """
        self.spark = spark

    @abstractmethod
    def write_stream(
        self,
        df: "DataFrame",
        checkpoint_path: str,
        **kwargs
    ) -> "StreamingQuery":
        """
        Write streaming DataFrame to sink.

        Args:
            df: Streaming DataFrame
            checkpoint_path: Checkpoint location

        Returns:
            StreamingQuery handle
        """
        pass


class KafkaSink(StreamSink):
    """
    Kafka stream sink.

    Writes streaming data to Apache Kafka topics.

    Example:
        >>> sink = KafkaSink(spark, "kafka:9092", "output-events")
        >>>
        >>> # Prepare output (must have 'key' and 'value' columns)
        >>> output = stream.selectExpr(
        ...     "CAST(id AS STRING) AS key",
        ...     "to_json(struct(*)) AS value"
        ... )
        >>>
        >>> query = sink.write_stream(
        ...     output,
        ...     checkpoint_path="/checkpoints/kafka-output"
        ... )
    """

    def __init__(
        self,
        spark: "SparkSession",
        bootstrap_servers: str,
        topic: str
    ) -> None:
        """
        Initialize Kafka sink.

        Args:
            spark: SparkSession
            bootstrap_servers: Kafka bootstrap servers
            topic: Target topic name
        """
        super().__init__(spark)
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic

    def write_stream(
        self,
        df: "DataFrame",
        checkpoint_path: str,
        output_mode: str = "append",
        trigger_interval: Optional[str] = None,
        query_name: Optional[str] = None
    ) -> "StreamingQuery":
        """
        Write streaming DataFrame to Kafka.

        Args:
            df: Streaming DataFrame (must have 'key' and 'value' columns)
            checkpoint_path: Checkpoint location
            output_mode: Output mode (append, update, complete)
            trigger_interval: Trigger interval (e.g., "10 seconds")
            query_name: Optional query name

        Returns:
            StreamingQuery handle
        """
        writer = (
            df.writeStream
            .format("kafka")
            .option("kafka.bootstrap.servers", self.bootstrap_servers)
            .option("topic", self.topic)
            .option("checkpointLocation", checkpoint_path)
            .outputMode(output_mode)
        )

        if trigger_interval:
            writer = writer.trigger(processingTime=trigger_interval)

        if query_name:
            writer = writer.queryName(query_name)

        logger.info(f"Starting Kafka stream to topic: {self.topic}")
        return writer.start()


class DeltaSink(StreamSink):
    """
    Delta Lake stream sink.

    Writes streaming data to Delta Lake tables with exactly-once semantics.

    Example:
        >>> # Append mode
        >>> sink = DeltaSink(spark, "catalog.schema.events")
        >>> query = sink.write_stream(
        ...     stream,
        ...     checkpoint_path="/checkpoints/events"
        ... )
        >>>
        >>> # With merge (upsert)
        >>> sink = DeltaSink(spark, "catalog.schema.events")
        >>> query = sink.write_stream_merge(
        ...     stream,
        ...     checkpoint_path="/checkpoints/events",
        ...     merge_keys=["id"]
        ... )
    """

    def __init__(
        self,
        spark: "SparkSession",
        table_name: str
    ) -> None:
        """
        Initialize Delta sink.

        Args:
            spark: SparkSession
            table_name: Delta table name (catalog.schema.table)
        """
        super().__init__(spark)
        self.table_name = table_name

    def write_stream(
        self,
        df: "DataFrame",
        checkpoint_path: str,
        output_mode: str = "append",
        trigger_interval: Optional[str] = None,
        query_name: Optional[str] = None,
        partition_by: Optional[list] = None
    ) -> "StreamingQuery":
        """
        Write streaming DataFrame to Delta table.

        Args:
            df: Streaming DataFrame
            checkpoint_path: Checkpoint location
            output_mode: Output mode (append, update, complete)
            trigger_interval: Trigger interval
            query_name: Optional query name
            partition_by: Optional partition columns

        Returns:
            StreamingQuery handle
        """
        writer = (
            df.writeStream
            .format("delta")
            .option("checkpointLocation", checkpoint_path)
            .outputMode(output_mode)
        )

        if trigger_interval:
            writer = writer.trigger(processingTime=trigger_interval)

        if query_name:
            writer = writer.queryName(query_name)

        if partition_by:
            writer = writer.partitionBy(*partition_by)

        logger.info(f"Starting Delta stream to table: {self.table_name}")
        return writer.toTable(self.table_name)

    def write_stream_merge(
        self,
        df: "DataFrame",
        checkpoint_path: str,
        merge_keys: list,
        trigger_interval: Optional[str] = None,
        query_name: Optional[str] = None,
        when_matched_update_all: bool = True,
        when_not_matched_insert_all: bool = True
    ) -> "StreamingQuery":
        """
        Write streaming DataFrame to Delta with merge (upsert).

        Uses foreachBatch to perform merge operations.

        Args:
            df: Streaming DataFrame
            checkpoint_path: Checkpoint location
            merge_keys: Columns to use for merge condition
            trigger_interval: Trigger interval
            query_name: Optional query name
            when_matched_update_all: Update all columns on match
            when_not_matched_insert_all: Insert all columns on no match

        Returns:
            StreamingQuery handle
        """
        from delta.tables import DeltaTable

        table_name = self.table_name

        def merge_batch(batch_df, batch_id):
            if batch_df.count() == 0:
                return

            delta_table = DeltaTable.forName(batch_df.sparkSession, table_name)

            # Build merge condition
            condition = " AND ".join(
                f"target.{key} = source.{key}" for key in merge_keys
            )

            merge_builder = (
                delta_table.alias("target")
                .merge(batch_df.alias("source"), condition)
            )

            if when_matched_update_all:
                merge_builder = merge_builder.whenMatchedUpdateAll()

            if when_not_matched_insert_all:
                merge_builder = merge_builder.whenNotMatchedInsertAll()

            merge_builder.execute()

        writer = (
            df.writeStream
            .foreachBatch(merge_batch)
            .option("checkpointLocation", checkpoint_path)
        )

        if trigger_interval:
            writer = writer.trigger(processingTime=trigger_interval)

        if query_name:
            writer = writer.queryName(query_name)

        logger.info(f"Starting Delta merge stream to table: {self.table_name}")
        return writer.start()


class FileSink(StreamSink):
    """
    File stream sink.

    Writes streaming data to files.

    Example:
        >>> sink = FileSink(spark, "s3://bucket/output", "parquet")
        >>> query = sink.write_stream(
        ...     stream,
        ...     checkpoint_path="/checkpoints/files",
        ...     partition_by=["date"]
        ... )
    """

    def __init__(
        self,
        spark: "SparkSession",
        path: str,
        format: str = "parquet"
    ) -> None:
        """
        Initialize file sink.

        Args:
            spark: SparkSession
            path: Output path
            format: File format (parquet, json, csv, orc)
        """
        super().__init__(spark)
        self.path = path
        self.format = format

    def write_stream(
        self,
        df: "DataFrame",
        checkpoint_path: str,
        output_mode: str = "append",
        trigger_interval: Optional[str] = None,
        query_name: Optional[str] = None,
        partition_by: Optional[list] = None,
        options: Optional[Dict[str, str]] = None
    ) -> "StreamingQuery":
        """
        Write streaming DataFrame to files.

        Args:
            df: Streaming DataFrame
            checkpoint_path: Checkpoint location
            output_mode: Output mode
            trigger_interval: Trigger interval
            query_name: Optional query name
            partition_by: Optional partition columns
            options: Additional format options

        Returns:
            StreamingQuery handle
        """
        writer = (
            df.writeStream
            .format(self.format)
            .option("checkpointLocation", checkpoint_path)
            .option("path", self.path)
            .outputMode(output_mode)
        )

        if trigger_interval:
            writer = writer.trigger(processingTime=trigger_interval)

        if query_name:
            writer = writer.queryName(query_name)

        if partition_by:
            writer = writer.partitionBy(*partition_by)

        if options:
            for key, value in options.items():
                writer = writer.option(key, value)

        logger.info(f"Starting file stream to: {self.path}")
        return writer.start()


class ConsoleSink(StreamSink):
    """
    Console stream sink for debugging.

    Outputs streaming data to console.

    Example:
        >>> sink = ConsoleSink(spark)
        >>> query = sink.write_stream(stream, truncate=False)
    """

    def __init__(self, spark: "SparkSession") -> None:
        """Initialize console sink."""
        super().__init__(spark)

    def write_stream(
        self,
        df: "DataFrame",
        checkpoint_path: str = "/tmp/console_checkpoint",
        output_mode: str = "append",
        truncate: bool = True,
        num_rows: int = 20,
        trigger_interval: Optional[str] = None
    ) -> "StreamingQuery":
        """
        Write streaming DataFrame to console.

        Args:
            df: Streaming DataFrame
            checkpoint_path: Checkpoint location (required but less important)
            output_mode: Output mode
            truncate: Truncate long strings
            num_rows: Number of rows to display
            trigger_interval: Trigger interval

        Returns:
            StreamingQuery handle
        """
        writer = (
            df.writeStream
            .format("console")
            .option("checkpointLocation", checkpoint_path)
            .option("truncate", str(truncate).lower())
            .option("numRows", num_rows)
            .outputMode(output_mode)
        )

        if trigger_interval:
            writer = writer.trigger(processingTime=trigger_interval)

        logger.info("Starting console stream output")
        return writer.start()


class ForeachBatchSink(StreamSink):
    """
    Custom batch processing sink.

    Allows custom processing of each micro-batch.

    Example:
        >>> def process_batch(batch_df, batch_id):
        ...     # Custom processing
        ...     batch_df.write.mode("append").saveAsTable("my_table")
        ...     # Additional processing
        ...     send_notifications(batch_df.count())
        >>>
        >>> sink = ForeachBatchSink(spark, process_batch)
        >>> query = sink.write_stream(stream, "/checkpoints/custom")
    """

    def __init__(
        self,
        spark: "SparkSession",
        batch_function: Callable[["DataFrame", int], None]
    ) -> None:
        """
        Initialize foreach batch sink.

        Args:
            spark: SparkSession
            batch_function: Function to process each batch
                           Signature: (batch_df: DataFrame, batch_id: int) -> None
        """
        super().__init__(spark)
        self.batch_function = batch_function

    def write_stream(
        self,
        df: "DataFrame",
        checkpoint_path: str,
        trigger_interval: Optional[str] = None,
        query_name: Optional[str] = None
    ) -> "StreamingQuery":
        """
        Write streaming DataFrame using custom batch function.

        Args:
            df: Streaming DataFrame
            checkpoint_path: Checkpoint location
            trigger_interval: Trigger interval
            query_name: Optional query name

        Returns:
            StreamingQuery handle
        """
        writer = (
            df.writeStream
            .foreachBatch(self.batch_function)
            .option("checkpointLocation", checkpoint_path)
        )

        if trigger_interval:
            writer = writer.trigger(processingTime=trigger_interval)

        if query_name:
            writer = writer.queryName(query_name)

        logger.info("Starting foreachBatch stream")
        return writer.start()


class MemorySink(StreamSink):
    """
    Memory stream sink for testing.

    Stores streaming data in memory for interactive queries.

    Warning: For testing only. Data is lost on restart.

    Example:
        >>> sink = MemorySink(spark, "my_stream")
        >>> query = sink.write_stream(stream)
        >>>
        >>> # Query results
        >>> spark.sql("SELECT * FROM my_stream").show()
    """

    def __init__(self, spark: "SparkSession", table_name: str) -> None:
        """
        Initialize memory sink.

        Args:
            spark: SparkSession
            table_name: In-memory table name for queries
        """
        super().__init__(spark)
        self.table_name = table_name

    def write_stream(
        self,
        df: "DataFrame",
        checkpoint_path: str = "/tmp/memory_checkpoint",
        output_mode: str = "append",
        trigger_interval: Optional[str] = None
    ) -> "StreamingQuery":
        """
        Write streaming DataFrame to memory.

        Args:
            df: Streaming DataFrame
            checkpoint_path: Checkpoint location
            output_mode: Output mode
            trigger_interval: Trigger interval

        Returns:
            StreamingQuery handle
        """
        logger.warning("Memory sink is for testing only. Data is lost on restart.")

        writer = (
            df.writeStream
            .format("memory")
            .queryName(self.table_name)
            .option("checkpointLocation", checkpoint_path)
            .outputMode(output_mode)
        )

        if trigger_interval:
            writer = writer.trigger(processingTime=trigger_interval)

        logger.info(f"Starting memory stream: {self.table_name}")
        return writer.start()
