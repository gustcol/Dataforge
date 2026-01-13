"""
DataForge Spark Engine Implementation

This module provides a PySpark-based implementation of the DataFrameEngine interface.
It is optimized for distributed processing of large datasets.

Performance Optimizations Applied:
    1. Adaptive Query Execution (AQE): Dynamic optimization at runtime
    2. Broadcast Joins: Automatic broadcast for small tables
    3. Partition Optimization: Configurable shuffle partitions
    4. Predicate Pushdown: Push filters to data source
    5. Column Pruning: Read only required columns
    6. Cache Management: Intelligent caching with automatic cleanup

Recommended Use Cases:
    - Datasets > 1GB
    - Distributed cluster processing
    - Production ETL pipelines
    - Delta Lake workloads
    - SQL-heavy analytics

Databricks Optimizations:
    - Photon runtime support
    - Delta Lake optimized writes
    - Auto-compaction
    - Z-ordering for query performance

Example:
    >>> from dataforge.engines import SparkEngine
    >>> from dataforge.core import SparkConfig
    >>>
    >>> config = SparkConfig(
    ...     shuffle_partitions=200,
    ...     adaptive_enabled=True,
    ...     photon_enabled=True  # Databricks only
    ... )
    >>> engine = SparkEngine(config=config)
    >>>
    >>> df = engine.read_parquet("s3://bucket/data/")
    >>> result = engine.groupby(df, ["region"], {"sales": "sum"})
    >>> engine.write_delta(result, "s3://bucket/output/")
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union, TYPE_CHECKING

from dataforge.core.base import (
    DataFrameEngine,
    EngineType,
    FileFormat,
    JoinType,
    ReadOptions,
    WriteOptions,
)
from dataforge.core.config import SparkConfig, DataForgeConfig
from dataforge.core.exceptions import (
    DataForgeError,
    EngineNotAvailableError,
    TransformationError,
)

if TYPE_CHECKING:
    import pandas as pd
    from pyspark.sql import DataFrame as SparkDataFrame, SparkSession


class SparkEngine(DataFrameEngine["SparkDataFrame"]):
    """
    PySpark implementation of DataFrameEngine.

    Provides distributed data processing with automatic application of
    Spark best practices and optimizations.

    Attributes:
        config: SparkConfig instance with engine settings
        spark: SparkSession instance

    Best Practices Applied:
        1. Adaptive Query Execution enabled by default
        2. Broadcast joins for small tables
        3. Optimized shuffle partitions
        4. Delta Lake integration
        5. Proper cache management

    Example:
        >>> engine = SparkEngine()
        >>> df = engine.read_parquet("hdfs://data/")
        >>> filtered = engine.filter(df, "amount > 1000")
        >>> engine.write_delta(filtered, "hdfs://output/")
    """

    def __init__(
        self,
        config: Optional[SparkConfig] = None,
        spark: Optional["SparkSession"] = None
    ) -> None:
        """
        Initialize SparkEngine.

        Args:
            config: Optional SparkConfig. Uses defaults if not provided.
            spark: Optional existing SparkSession. Creates new if not provided.
        """
        self.config = config or SparkConfig()
        self._spark = spark
        self._cached_dfs: List["SparkDataFrame"] = []

    @property
    def spark(self) -> "SparkSession":
        """
        Get or create SparkSession.

        The session is configured with optimizations from SparkConfig.
        """
        if self._spark is None:
            self._spark = self._create_spark_session()
        return self._spark

    def _create_spark_session(self) -> "SparkSession":
        """Create and configure SparkSession."""
        try:
            from pyspark.sql import SparkSession
        except ImportError:
            raise EngineNotAvailableError(
                "spark",
                "PySpark is not installed",
                "pip install pyspark"
            )

        builder = SparkSession.builder.appName("DataForge")

        # Apply configuration
        spark_conf = self.config.to_spark_conf()
        for key, value in spark_conf.items():
            builder = builder.config(key, value)

        # Try to get existing session or create new
        spark = builder.getOrCreate()

        return spark

    @property
    def engine_type(self) -> EngineType:
        """Return engine type identifier."""
        return EngineType.SPARK

    @property
    def is_available(self) -> bool:
        """Check if Spark is available."""
        return self.check_availability()

    @staticmethod
    def check_availability() -> bool:
        """Static method to check Spark availability."""
        try:
            from pyspark.sql import SparkSession
            return True
        except ImportError:
            return False

    def _convert_read_options(
        self,
        options: Optional[ReadOptions],
        file_format: FileFormat
    ) -> Dict[str, Any]:
        """Convert ReadOptions to Spark DataFrameReader options."""
        if options is None:
            options = ReadOptions()

        spark_options: Dict[str, Any] = {}

        if file_format == FileFormat.CSV:
            spark_options["header"] = str(options.header).lower()
            spark_options["inferSchema"] = str(options.infer_schema).lower()
            spark_options["encoding"] = options.encoding
            spark_options["sep"] = options.delimiter
            spark_options["nullValue"] = options.null_values[0] if options.null_values else ""
            spark_options["dateFormat"] = options.date_format
            spark_options["timestampFormat"] = options.timestamp_format

        elif file_format == FileFormat.JSON:
            spark_options["multiLine"] = str(options.multiline).lower()

        return spark_options

    def _apply_read_optimizations(
        self,
        df: "SparkDataFrame",
        options: Optional[ReadOptions]
    ) -> "SparkDataFrame":
        """Apply post-read optimizations."""
        if options is None:
            return df

        # Column selection (projection pushdown)
        if options.columns:
            df = df.select(*options.columns)

        # Filter (predicate pushdown)
        if options.filter:
            df = df.filter(options.filter)

        # Sampling
        if options.sample_fraction:
            df = df.sample(fraction=options.sample_fraction, seed=42)

        return df

    # ==========================================================================
    # READ OPERATIONS
    # ==========================================================================

    def read_csv(
        self,
        path: Union[str, Path, List[str]],
        options: Optional[ReadOptions] = None
    ) -> "SparkDataFrame":
        """
        Read CSV file(s) into a Spark DataFrame.

        Supports reading from:
            - Local filesystem
            - HDFS
            - S3 (s3a://)
            - Azure Blob (wasbs://)
            - Google Cloud Storage (gs://)

        Args:
            path: Path to CSV file(s). Supports glob patterns.
            options: Read configuration options

        Returns:
            Spark DataFrame

        Performance Tips:
            - Provide explicit schema to avoid inference scan
            - Use partition discovery for partitioned data
        """
        spark_options = self._convert_read_options(options, FileFormat.CSV)

        reader = self.spark.read.format("csv").options(**spark_options)

        # Apply schema if provided
        if options and options.schema:
            from pyspark.sql.types import StructType
            schema = self._convert_schema_to_spark(options.schema)
            reader = reader.schema(schema)

        # Handle path types
        if isinstance(path, list):
            df = reader.load(path)
        else:
            df = reader.load(str(path))

        return self._apply_read_optimizations(df, options)

    def read_parquet(
        self,
        path: Union[str, Path, List[str]],
        options: Optional[ReadOptions] = None
    ) -> "SparkDataFrame":
        """
        Read Parquet file(s) into a Spark DataFrame.

        Parquet is the recommended format for Spark workloads due to:
            - Columnar storage for efficient analytics
            - Predicate pushdown support
            - Schema preservation
            - Splittable for parallel reads

        Args:
            path: Path to Parquet file(s) or directory
            options: Read configuration options

        Returns:
            Spark DataFrame
        """
        if isinstance(path, list):
            df = self.spark.read.parquet(*path)
        else:
            df = self.spark.read.parquet(str(path))

        return self._apply_read_optimizations(df, options)

    def read_json(
        self,
        path: Union[str, Path, List[str]],
        options: Optional[ReadOptions] = None
    ) -> "SparkDataFrame":
        """
        Read JSON file(s) into a Spark DataFrame.

        Args:
            path: Path to JSON file(s)
            options: Read configuration options

        Returns:
            Spark DataFrame
        """
        spark_options = self._convert_read_options(options, FileFormat.JSON)
        reader = self.spark.read.format("json").options(**spark_options)

        if isinstance(path, list):
            df = reader.load(path)
        else:
            df = reader.load(str(path))

        return self._apply_read_optimizations(df, options)

    def read_delta(
        self,
        path: Union[str, Path],
        options: Optional[ReadOptions] = None,
        version: Optional[int] = None,
        timestamp: Optional[str] = None
    ) -> "SparkDataFrame":
        """
        Read Delta Lake table into a Spark DataFrame.

        Delta Lake provides:
            - ACID transactions
            - Time travel (versioning)
            - Schema evolution
            - Unified batch and streaming

        Args:
            path: Path to Delta table
            options: Read configuration options
            version: Specific version to read (time travel)
            timestamp: Timestamp for time travel query

        Returns:
            Spark DataFrame

        Example:
            >>> # Read latest
            >>> df = engine.read_delta("s3://bucket/delta_table/")
            >>>
            >>> # Time travel by version
            >>> df = engine.read_delta("s3://bucket/delta_table/", version=10)
            >>>
            >>> # Time travel by timestamp
            >>> df = engine.read_delta(
            ...     "s3://bucket/delta_table/",
            ...     timestamp="2024-01-01"
            ... )
        """
        reader = self.spark.read.format("delta")

        if version is not None:
            reader = reader.option("versionAsOf", version)
        elif timestamp is not None:
            reader = reader.option("timestampAsOf", timestamp)

        df = reader.load(str(path))

        return self._apply_read_optimizations(df, options)

    def _convert_schema_to_spark(self, schema: Dict[str, str]) -> "StructType":
        """Convert schema dictionary to Spark StructType."""
        from pyspark.sql.types import (
            StructType, StructField, StringType, IntegerType, LongType,
            FloatType, DoubleType, BooleanType, DateType, TimestampType
        )

        type_mapping = {
            "string": StringType(),
            "str": StringType(),
            "int": IntegerType(),
            "int32": IntegerType(),
            "int64": LongType(),
            "long": LongType(),
            "float": FloatType(),
            "float32": FloatType(),
            "float64": DoubleType(),
            "double": DoubleType(),
            "bool": BooleanType(),
            "boolean": BooleanType(),
            "date": DateType(),
            "timestamp": TimestampType(),
            "datetime": TimestampType(),
        }

        fields = []
        for col_name, col_type in schema.items():
            spark_type = type_mapping.get(col_type.lower(), StringType())
            fields.append(StructField(col_name, spark_type, True))

        return StructType(fields)

    # ==========================================================================
    # WRITE OPERATIONS
    # ==========================================================================

    def write_csv(
        self,
        df: "SparkDataFrame",
        path: Union[str, Path],
        options: Optional[WriteOptions] = None
    ) -> None:
        """
        Write DataFrame to CSV file(s).

        Args:
            df: DataFrame to write
            path: Output path
            options: Write configuration options
        """
        if options is None:
            options = WriteOptions()

        writer = df.write.format("csv").mode(options.mode)

        if options.header:
            writer = writer.option("header", "true")

        if options.compression:
            writer = writer.option("compression", options.compression)

        if options.partition_by:
            writer = writer.partitionBy(*options.partition_by)

        if options.coalesce:
            df = df.coalesce(options.coalesce)
            writer = df.write.format("csv").mode(options.mode)

        writer.save(str(path))

    def write_parquet(
        self,
        df: "SparkDataFrame",
        path: Union[str, Path],
        options: Optional[WriteOptions] = None
    ) -> None:
        """
        Write DataFrame to Parquet file(s).

        Args:
            df: DataFrame to write
            path: Output path
            options: Write configuration options
        """
        if options is None:
            options = WriteOptions()

        writer = df.write.format("parquet").mode(options.mode)

        if options.compression:
            writer = writer.option("compression", options.compression)

        if options.partition_by:
            writer = writer.partitionBy(*options.partition_by)

        if options.coalesce:
            df = df.coalesce(options.coalesce)
            writer = df.write.format("parquet").mode(options.mode)

        writer.save(str(path))

    def write_delta(
        self,
        df: "SparkDataFrame",
        path: Union[str, Path],
        options: Optional[WriteOptions] = None
    ) -> None:
        """
        Write DataFrame to Delta Lake table.

        Args:
            df: DataFrame to write
            path: Output path
            options: Write configuration options

        Post-write Optimizations (when options.optimize=True):
            - Runs OPTIMIZE to compact small files
            - Applies Z-ORDER if z_order_by specified
        """
        if options is None:
            options = WriteOptions()

        writer = df.write.format("delta").mode(options.mode)

        if options.partition_by:
            writer = writer.partitionBy(*options.partition_by)

        if options.merge_schema:
            writer = writer.option("mergeSchema", "true")

        if options.coalesce:
            df = df.coalesce(options.coalesce)
            writer = df.write.format("delta").mode(options.mode)

        writer.save(str(path))

        # Post-write optimizations
        if options.optimize:
            self._optimize_delta_table(str(path), options.z_order_by)

    def _optimize_delta_table(
        self,
        path: str,
        z_order_by: Optional[List[str]] = None
    ) -> None:
        """Run OPTIMIZE on Delta table."""
        from delta.tables import DeltaTable

        delta_table = DeltaTable.forPath(self.spark, path)

        if z_order_by:
            delta_table.optimize().executeZOrderBy(*z_order_by)
        else:
            delta_table.optimize().executeCompaction()

    # ==========================================================================
    # TRANSFORMATION OPERATIONS
    # ==========================================================================

    def filter(
        self,
        df: "SparkDataFrame",
        condition: str
    ) -> "SparkDataFrame":
        """
        Filter rows using SQL expression.

        Uses Spark's Catalyst optimizer for predicate pushdown.

        Args:
            df: Input DataFrame
            condition: SQL filter expression

        Returns:
            Filtered DataFrame
        """
        try:
            return df.filter(condition)
        except Exception as e:
            raise TransformationError(
                "filter",
                f"Failed to apply filter: {condition}",
                condition,
                e
            )

    def select(
        self,
        df: "SparkDataFrame",
        columns: List[str]
    ) -> "SparkDataFrame":
        """Select specific columns."""
        return df.select(*columns)

    def rename(
        self,
        df: "SparkDataFrame",
        columns: Dict[str, str]
    ) -> "SparkDataFrame":
        """Rename columns."""
        result = df
        for old_name, new_name in columns.items():
            result = result.withColumnRenamed(old_name, new_name)
        return result

    def with_column(
        self,
        df: "SparkDataFrame",
        name: str,
        expression: Union[str, Callable]
    ) -> "SparkDataFrame":
        """
        Add or replace a column.

        Args:
            df: Input DataFrame
            name: Column name
            expression: SQL expression string or Column expression

        Returns:
            DataFrame with new column
        """
        from pyspark.sql import functions as F

        if isinstance(expression, str):
            return df.withColumn(name, F.expr(expression))
        else:
            return df.withColumn(name, expression(df))

    def drop(
        self,
        df: "SparkDataFrame",
        columns: List[str]
    ) -> "SparkDataFrame":
        """Drop columns from DataFrame."""
        return df.drop(*columns)

    def distinct(self, df: "SparkDataFrame") -> "SparkDataFrame":
        """Return distinct rows."""
        return df.distinct()

    def sort(
        self,
        df: "SparkDataFrame",
        columns: List[str],
        ascending: Union[bool, List[bool]] = True
    ) -> "SparkDataFrame":
        """
        Sort DataFrame by columns.

        Note: Sorting is expensive in Spark as it requires shuffle.
        Consider if sorting is necessary for your use case.
        """
        if isinstance(ascending, bool):
            ascending = [ascending] * len(columns)

        return df.orderBy(
            *[
                df[col].asc() if asc else df[col].desc()
                for col, asc in zip(columns, ascending)
            ]
        )

    def limit(self, df: "SparkDataFrame", n: int) -> "SparkDataFrame":
        """Return first n rows."""
        return df.limit(n)

    # ==========================================================================
    # AGGREGATION OPERATIONS
    # ==========================================================================

    def groupby(
        self,
        df: "SparkDataFrame",
        columns: List[str],
        aggregations: Dict[str, Union[str, List[str]]]
    ) -> "SparkDataFrame":
        """
        Group by columns and apply aggregations.

        Uses Spark's native aggregation functions for optimal performance.
        Avoid UDFs for aggregations - they are significantly slower.

        Args:
            df: Input DataFrame
            columns: Group by columns
            aggregations: Column to aggregation(s) mapping

        Returns:
            Aggregated DataFrame
        """
        from pyspark.sql import functions as F

        agg_funcs = {
            "sum": F.sum,
            "avg": F.avg,
            "mean": F.mean,
            "min": F.min,
            "max": F.max,
            "count": F.count,
            "first": F.first,
            "last": F.last,
            "std": F.stddev,
            "stddev": F.stddev,
            "var": F.variance,
            "variance": F.variance,
            "collect_list": F.collect_list,
            "collect_set": F.collect_set,
        }

        agg_exprs = []
        for col, aggs in aggregations.items():
            if isinstance(aggs, str):
                aggs = [aggs]
            for agg in aggs:
                agg_func = agg_funcs.get(agg.lower())
                if agg_func:
                    alias = f"{col}_{agg}" if len(aggs) > 1 or len(aggregations) > 1 else col
                    agg_exprs.append(agg_func(col).alias(alias))

        return df.groupBy(*columns).agg(*agg_exprs)

    def agg(
        self,
        df: "SparkDataFrame",
        aggregations: Dict[str, Union[str, List[str]]]
    ) -> "SparkDataFrame":
        """Apply aggregations without grouping."""
        from pyspark.sql import functions as F

        agg_funcs = {
            "sum": F.sum,
            "avg": F.avg,
            "mean": F.mean,
            "min": F.min,
            "max": F.max,
            "count": F.count,
            "std": F.stddev,
            "var": F.variance,
        }

        agg_exprs = []
        for col, aggs in aggregations.items():
            if isinstance(aggs, str):
                aggs = [aggs]
            for agg in aggs:
                agg_func = agg_funcs.get(agg.lower())
                if agg_func:
                    agg_exprs.append(agg_func(col).alias(f"{col}_{agg}"))

        return df.agg(*agg_exprs)

    # ==========================================================================
    # JOIN OPERATIONS
    # ==========================================================================

    def join(
        self,
        left: "SparkDataFrame",
        right: "SparkDataFrame",
        on: Optional[Union[str, List[str]]] = None,
        left_on: Optional[Union[str, List[str]]] = None,
        right_on: Optional[Union[str, List[str]]] = None,
        how: JoinType = JoinType.INNER
    ) -> "SparkDataFrame":
        """
        Join two DataFrames.

        Spark automatically chooses join strategy:
            - Broadcast join: For small right tables (< broadcast threshold)
            - Sort-merge join: For large tables with sorted keys
            - Shuffle hash join: General case

        Performance Tips:
            - Broadcast small dimension tables explicitly: broadcast(df)
            - Ensure join keys have same types
            - Filter before joining to reduce shuffle

        Args:
            left: Left DataFrame
            right: Right DataFrame
            on: Join column(s) when same in both
            left_on: Left join column(s)
            right_on: Right join column(s)
            how: Join type

        Returns:
            Joined DataFrame
        """
        join_map = {
            JoinType.INNER: "inner",
            JoinType.LEFT: "left",
            JoinType.RIGHT: "right",
            JoinType.OUTER: "outer",
            JoinType.CROSS: "cross",
            JoinType.LEFT_SEMI: "left_semi",
            JoinType.LEFT_ANTI: "left_anti",
        }

        spark_how = join_map.get(how, "inner")

        if on:
            return left.join(right, on=on, how=spark_how)
        elif left_on and right_on:
            # Handle different column names
            left_cols = [left_on] if isinstance(left_on, str) else left_on
            right_cols = [right_on] if isinstance(right_on, str) else right_on

            condition = None
            for l_col, r_col in zip(left_cols, right_cols):
                cond = left[l_col] == right[r_col]
                condition = cond if condition is None else condition & cond

            return left.join(right, on=condition, how=spark_how)
        else:
            raise TransformationError(
                "join",
                "Must specify either 'on' or both 'left_on' and 'right_on'"
            )

    def broadcast_join(
        self,
        left: "SparkDataFrame",
        right: "SparkDataFrame",
        on: Union[str, List[str]],
        how: JoinType = JoinType.INNER
    ) -> "SparkDataFrame":
        """
        Perform broadcast join (explicitly broadcast right table).

        Use when right table is small (< 100MB recommended).
        Avoids shuffle of left table - significant performance gain.

        Args:
            left: Left (large) DataFrame
            right: Right (small) DataFrame to broadcast
            on: Join column(s)
            how: Join type

        Returns:
            Joined DataFrame
        """
        from pyspark.sql.functions import broadcast

        join_map = {
            JoinType.INNER: "inner",
            JoinType.LEFT: "left",
            JoinType.RIGHT: "right",
        }

        return left.join(broadcast(right), on=on, how=join_map.get(how, "inner"))

    # ==========================================================================
    # UTILITY OPERATIONS
    # ==========================================================================

    def count(self, df: "SparkDataFrame") -> int:
        """Count number of rows."""
        return df.count()

    def columns(self, df: "SparkDataFrame") -> List[str]:
        """Get column names."""
        return df.columns

    def dtypes(self, df: "SparkDataFrame") -> Dict[str, str]:
        """Get column data types."""
        return {col: dtype for col, dtype in df.dtypes}

    def schema(self, df: "SparkDataFrame") -> Dict[str, Any]:
        """Get full schema information."""
        return {
            "columns": df.columns,
            "dtypes": dict(df.dtypes),
            "schema_json": df.schema.json(),
        }

    def to_pandas(self, df: "SparkDataFrame") -> "pd.DataFrame":
        """
        Convert to pandas DataFrame.

        WARNING: Collects all data to driver. Use with caution on large datasets.
        Consider using .limit() first or sampling.
        """
        return df.toPandas()

    def from_pandas(self, df: "pd.DataFrame") -> "SparkDataFrame":
        """Create Spark DataFrame from pandas DataFrame."""
        return self.spark.createDataFrame(df)

    def cache(self, df: "SparkDataFrame") -> "SparkDataFrame":
        """
        Cache DataFrame in memory.

        Uses storage level from config (default: MEMORY_AND_DISK).
        Remember to uncache when done to free resources.
        """
        from pyspark import StorageLevel

        storage_levels = {
            "MEMORY_ONLY": StorageLevel.MEMORY_ONLY,
            "MEMORY_AND_DISK": StorageLevel.MEMORY_AND_DISK,
            "DISK_ONLY": StorageLevel.DISK_ONLY,
            "MEMORY_ONLY_SER": StorageLevel.MEMORY_ONLY_SER,
            "MEMORY_AND_DISK_SER": StorageLevel.MEMORY_AND_DISK_SER,
        }

        level = storage_levels.get(
            self.config.cache_storage_level,
            StorageLevel.MEMORY_AND_DISK
        )

        cached = df.persist(level)
        self._cached_dfs.append(cached)
        return cached

    def uncache(self, df: "SparkDataFrame") -> "SparkDataFrame":
        """Remove DataFrame from cache."""
        df.unpersist()
        if df in self._cached_dfs:
            self._cached_dfs.remove(df)
        return df

    def uncache_all(self) -> None:
        """Uncache all DataFrames cached by this engine."""
        for df in self._cached_dfs:
            try:
                df.unpersist()
            except Exception:
                pass
        self._cached_dfs.clear()

    def show(
        self,
        df: "SparkDataFrame",
        n: int = 20,
        truncate: bool = True
    ) -> None:
        """Display DataFrame rows."""
        df.show(n=n, truncate=truncate)

    def collect(self, df: "SparkDataFrame") -> List[Dict[str, Any]]:
        """
        Collect all rows as list of dictionaries.

        WARNING: Collects all data to driver memory.
        """
        return [row.asDict() for row in df.collect()]

    def head(self, df: "SparkDataFrame", n: int = 5) -> List[Dict[str, Any]]:
        """Get first n rows as dictionaries."""
        return [row.asDict() for row in df.take(n)]

    def sql(self, query: str) -> "SparkDataFrame":
        """
        Execute SQL query.

        Args:
            query: SQL query string

        Returns:
            Result DataFrame

        Example:
            >>> df.createOrReplaceTempView("my_table")
            >>> result = engine.sql("SELECT * FROM my_table WHERE amount > 100")
        """
        return self.spark.sql(query)

    def explain(self, df: "SparkDataFrame", extended: bool = False) -> None:
        """
        Print query execution plan.

        Useful for understanding and optimizing query performance.

        Args:
            df: DataFrame to explain
            extended: Show extended plan with optimization steps
        """
        df.explain(extended=extended)
