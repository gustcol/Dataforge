"""
DataForge Stream Processors

Stream processing utilities for transformations, windowing,
and aggregations on streaming data.

Features:
    - Watermarking for late data handling
    - Window aggregations (tumbling, sliding, session)
    - Stateful processing patterns
    - Stream deduplication
    - Stream joins

Best Practices:
    1. Always use watermarks for aggregations
    2. Choose appropriate window sizes
    3. Consider state store memory usage
    4. Use dropDuplicates for exactly-once
    5. Monitor late data metrics

Example:
    >>> processor = StreamProcessor(stream)
    >>>
    >>> # Add watermark
    >>> processor.with_watermark("event_time", "10 minutes")
    >>>
    >>> # Window aggregation
    >>> result = processor.window_aggregate(
    ...     time_column="event_time",
    ...     window_duration="5 minutes",
    ...     aggregations={"value": "sum", "count": "count"}
    ... )
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union, TYPE_CHECKING
import logging

if TYPE_CHECKING:
    from pyspark.sql import DataFrame
    from pyspark.sql.streaming import StreamingQuery

logger = logging.getLogger(__name__)


@dataclass
class WatermarkConfig:
    """Watermark configuration.

    Attributes:
        column: Event time column name
        delay_threshold: Maximum delay threshold (e.g., "10 minutes")
    """
    column: str
    delay_threshold: str


@dataclass
class WindowConfig:
    """Window configuration.

    Attributes:
        time_column: Event time column
        window_duration: Window size (e.g., "5 minutes")
        slide_duration: Slide interval for sliding windows
        start_time: Optional start time offset
    """
    time_column: str
    window_duration: str
    slide_duration: Optional[str] = None
    start_time: Optional[str] = None


class StreamProcessor:
    """
    Stream processor for transformations and aggregations.

    Provides a fluent interface for common streaming operations.

    Example:
        >>> processor = StreamProcessor(stream)
        >>>
        >>> # Chain operations
        >>> result = (
        ...     processor
        ...     .with_watermark("timestamp", "10 minutes")
        ...     .filter("value > 0")
        ...     .select(["id", "value", "timestamp"])
        ...     .window_aggregate(
        ...         "timestamp", "5 minutes",
        ...         {"value": "sum"}
        ...     )
        ...     .stream
        ... )
    """

    def __init__(self, df: "DataFrame") -> None:
        """
        Initialize stream processor.

        Args:
            df: Streaming DataFrame
        """
        self._df = df
        self._watermark: Optional[WatermarkConfig] = None

    @property
    def stream(self) -> "DataFrame":
        """Get the underlying streaming DataFrame."""
        return self._df

    def with_watermark(
        self,
        column: str,
        delay_threshold: str
    ) -> "StreamProcessor":
        """
        Add watermark for handling late data.

        Watermarks define how late data can arrive and still be processed.
        Required for stateful operations like aggregations and joins.

        Args:
            column: Event time column name
            delay_threshold: Maximum lateness threshold (e.g., "10 minutes")

        Returns:
            Self for method chaining

        Example:
            >>> processor.with_watermark("event_time", "10 minutes")
        """
        self._watermark = WatermarkConfig(column, delay_threshold)
        self._df = self._df.withWatermark(column, delay_threshold)

        logger.info(f"Added watermark on '{column}' with threshold '{delay_threshold}'")
        return self

    def filter(self, condition: str) -> "StreamProcessor":
        """
        Filter streaming data.

        Args:
            condition: SQL filter condition

        Returns:
            Self for method chaining
        """
        self._df = self._df.filter(condition)
        return self

    def select(self, columns: List[str]) -> "StreamProcessor":
        """
        Select columns from stream.

        Args:
            columns: Column names to select

        Returns:
            Self for method chaining
        """
        self._df = self._df.select(*columns)
        return self

    def drop(self, columns: List[str]) -> "StreamProcessor":
        """
        Drop columns from stream.

        Args:
            columns: Column names to drop

        Returns:
            Self for method chaining
        """
        self._df = self._df.drop(*columns)
        return self

    def with_column(self, name: str, expression: Any) -> "StreamProcessor":
        """
        Add or replace a column.

        Args:
            name: Column name
            expression: Column expression (string or Column)

        Returns:
            Self for method chaining
        """
        from pyspark.sql import functions as F

        if isinstance(expression, str):
            expression = F.expr(expression)

        self._df = self._df.withColumn(name, expression)
        return self

    def rename_columns(self, mapping: Dict[str, str]) -> "StreamProcessor":
        """
        Rename columns.

        Args:
            mapping: Old name to new name mapping

        Returns:
            Self for method chaining
        """
        for old_name, new_name in mapping.items():
            self._df = self._df.withColumnRenamed(old_name, new_name)
        return self

    def window_aggregate(
        self,
        time_column: str,
        window_duration: str,
        aggregations: Dict[str, str],
        slide_duration: Optional[str] = None,
        group_by: Optional[List[str]] = None
    ) -> "StreamProcessor":
        """
        Perform windowed aggregation.

        Args:
            time_column: Event time column
            window_duration: Window size (e.g., "5 minutes", "1 hour")
            aggregations: Column to aggregation function mapping
                         e.g., {"value": "sum", "count": "count"}
            slide_duration: Slide interval for sliding windows
            group_by: Additional grouping columns

        Returns:
            Self for method chaining

        Example:
            >>> # Tumbling window (5 minute)
            >>> processor.window_aggregate(
            ...     "timestamp", "5 minutes",
            ...     {"value": "sum", "id": "count"}
            ... )
            >>>
            >>> # Sliding window (5 minute window, 1 minute slide)
            >>> processor.window_aggregate(
            ...     "timestamp", "5 minutes",
            ...     {"value": "avg"},
            ...     slide_duration="1 minute"
            ... )
        """
        from pyspark.sql import functions as F

        # Create window column
        if slide_duration:
            window_col = F.window(time_column, window_duration, slide_duration)
        else:
            window_col = F.window(time_column, window_duration)

        # Build grouping columns
        group_cols = [window_col]
        if group_by:
            group_cols.extend([F.col(c) for c in group_by])

        # Build aggregation expressions
        agg_exprs = []
        for col_name, agg_func in aggregations.items():
            if agg_func == "count":
                agg_exprs.append(F.count(col_name).alias(f"{col_name}_{agg_func}"))
            elif agg_func == "sum":
                agg_exprs.append(F.sum(col_name).alias(f"{col_name}_{agg_func}"))
            elif agg_func == "avg":
                agg_exprs.append(F.avg(col_name).alias(f"{col_name}_{agg_func}"))
            elif agg_func == "min":
                agg_exprs.append(F.min(col_name).alias(f"{col_name}_{agg_func}"))
            elif agg_func == "max":
                agg_exprs.append(F.max(col_name).alias(f"{col_name}_{agg_func}"))
            elif agg_func == "first":
                agg_exprs.append(F.first(col_name).alias(f"{col_name}_{agg_func}"))
            elif agg_func == "last":
                agg_exprs.append(F.last(col_name).alias(f"{col_name}_{agg_func}"))

        self._df = self._df.groupBy(*group_cols).agg(*agg_exprs)

        window_type = "sliding" if slide_duration else "tumbling"
        logger.info(f"Added {window_type} window aggregation: {window_duration}")
        return self

    def group_aggregate(
        self,
        group_by: List[str],
        aggregations: Dict[str, str]
    ) -> "StreamProcessor":
        """
        Perform grouped aggregation without windowing.

        Requires complete output mode.

        Args:
            group_by: Grouping columns
            aggregations: Column to aggregation function mapping

        Returns:
            Self for method chaining
        """
        from pyspark.sql import functions as F

        agg_exprs = []
        for col_name, agg_func in aggregations.items():
            if agg_func == "count":
                agg_exprs.append(F.count(col_name).alias(f"{col_name}_{agg_func}"))
            elif agg_func == "sum":
                agg_exprs.append(F.sum(col_name).alias(f"{col_name}_{agg_func}"))
            elif agg_func == "avg":
                agg_exprs.append(F.avg(col_name).alias(f"{col_name}_{agg_func}"))
            elif agg_func == "min":
                agg_exprs.append(F.min(col_name).alias(f"{col_name}_{agg_func}"))
            elif agg_func == "max":
                agg_exprs.append(F.max(col_name).alias(f"{col_name}_{agg_func}"))

        self._df = self._df.groupBy(*group_by).agg(*agg_exprs)
        return self

    def drop_duplicates(
        self,
        columns: Optional[List[str]] = None
    ) -> "StreamProcessor":
        """
        Drop duplicate records.

        For streaming, deduplication is performed within the watermark.

        Args:
            columns: Columns to consider for deduplication (None = all)

        Returns:
            Self for method chaining

        Example:
            >>> # Deduplicate by id column
            >>> processor.with_watermark("timestamp", "10 minutes")
            >>> processor.drop_duplicates(["id"])
        """
        if columns:
            self._df = self._df.dropDuplicates(columns)
        else:
            self._df = self._df.dropDuplicates()

        logger.info(f"Added deduplication on columns: {columns or 'all'}")
        return self

    def join(
        self,
        other: "DataFrame",
        on: Union[str, List[str]],
        how: str = "inner"
    ) -> "StreamProcessor":
        """
        Join with another DataFrame.

        Supports stream-stream and stream-static joins.

        Args:
            other: DataFrame to join with
            on: Join column(s)
            how: Join type (inner, left, right, outer)

        Returns:
            Self for method chaining

        Example:
            >>> # Stream-static join
            >>> processor.join(static_df, "customer_id", "left")
            >>>
            >>> # Stream-stream join (both must have watermarks)
            >>> processor.with_watermark("timestamp", "10 minutes")
            >>> processor.join(
            ...     other_stream.withWatermark("timestamp", "10 minutes"),
            ...     "id"
            ... )
        """
        self._df = self._df.join(other, on=on, how=how)

        logger.info(f"Added {how} join on {on}")
        return self

    def transform(
        self,
        func: Callable[["DataFrame"], "DataFrame"]
    ) -> "StreamProcessor":
        """
        Apply custom transformation function.

        Args:
            func: Function that takes and returns DataFrame

        Returns:
            Self for method chaining

        Example:
            >>> def add_features(df):
            ...     return df.withColumn("feature", F.col("a") + F.col("b"))
            >>>
            >>> processor.transform(add_features)
        """
        self._df = func(self._df)
        return self

    def parse_json(
        self,
        column: str,
        schema: Any,
        output_column: Optional[str] = None
    ) -> "StreamProcessor":
        """
        Parse JSON column.

        Args:
            column: Column containing JSON strings
            schema: Schema for JSON parsing
            output_column: Output column name (default: same as input)

        Returns:
            Self for method chaining
        """
        from pyspark.sql import functions as F

        output = output_column or column
        self._df = self._df.withColumn(output, F.from_json(F.col(column), schema))

        return self

    def flatten_json(self, column: str) -> "StreamProcessor":
        """
        Flatten a struct column to individual columns.

        Args:
            column: Struct column to flatten

        Returns:
            Self for method chaining
        """
        # Get struct field names
        struct_fields = [
            f.name for f in self._df.schema[column].dataType.fields
        ]

        # Select all columns plus flattened struct fields
        for field in struct_fields:
            self._df = self._df.withColumn(
                field,
                self._df[f"{column}.{field}"]
            )

        # Drop original struct column
        self._df = self._df.drop(column)

        return self

    def explain(self, extended: bool = False) -> None:
        """
        Print the streaming query plan.

        Args:
            extended: Include extended information
        """
        self._df.explain(extended)


def create_rate_limiter(
    max_records_per_second: int
) -> Callable[["DataFrame", int], "DataFrame"]:
    """
    Create a rate limiter function for foreachBatch.

    Args:
        max_records_per_second: Maximum throughput

    Returns:
        Function to use with foreachBatch

    Example:
        >>> rate_limited = create_rate_limiter(1000)
        >>> stream.writeStream.foreachBatch(rate_limited).start()
    """
    import time

    def rate_limit(batch_df: "DataFrame", batch_id: int) -> "DataFrame":
        count = batch_df.count()
        if count > 0:
            delay = count / max_records_per_second
            time.sleep(delay)
        return batch_df

    return rate_limit


def monitor_stream(query: "StreamingQuery") -> Dict[str, Any]:
    """
    Get monitoring information for a streaming query.

    Args:
        query: StreamingQuery to monitor

    Returns:
        Dictionary with monitoring metrics
    """
    status = query.status
    progress = query.lastProgress

    info = {
        "id": str(query.id),
        "name": query.name,
        "is_active": query.isActive,
        "status": {
            "message": status.get("message", ""),
            "is_data_available": status.get("isDataAvailable", False),
            "is_trigger_active": status.get("isTriggerActive", False),
        },
    }

    if progress:
        info["progress"] = {
            "batch_id": progress.get("batchId"),
            "num_input_rows": progress.get("numInputRows"),
            "input_rows_per_second": progress.get("inputRowsPerSecond"),
            "processed_rows_per_second": progress.get("processedRowsPerSecond"),
            "trigger_execution_ms": progress.get("triggerExecution", {}).get("latency"),
        }

    return info
