"""
DataForge Abstract Base Classes

This module defines the abstract interfaces that all engine implementations
must follow. This ensures consistent API across Pandas, Spark, and RAPIDS.

Classes:
    - DataFrameEngine: Abstract base for all DataFrame operations
    - ReadOptions: Configuration for read operations
    - WriteOptions: Configuration for write operations

The base classes use Python's ABC (Abstract Base Class) module to enforce
implementation of required methods in concrete engine classes.

Example:
    >>> from dataforge.core.base import DataFrameEngine
    >>>
    >>> class MyCustomEngine(DataFrameEngine):
    ...     def read_csv(self, path, options):
    ...         # Implementation
    ...         pass
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    Generic,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
    Callable,
    Iterator,
)
from enum import Enum
from pathlib import Path


# Type variable for the native DataFrame type
T = TypeVar("T")  # Native DataFrame type (pd.DataFrame, spark.DataFrame, cudf.DataFrame)


class EngineType(str, Enum):
    """
    Enumeration of supported processing engines.

    Used to specify which engine to use for data processing operations.
    AUTO enables intelligent engine selection based on data characteristics.

    Attributes:
        PANDAS: Single-node, in-memory processing using pandas
            Best for: < 1GB datasets, complex transformations, EDA

        POLARS: High-performance single-node processing using Polars
            Best for: 100MB - 10GB datasets, lazy evaluation, Rust-backed speed

        SPARK: Distributed processing using PySpark
            Best for: > 1GB datasets, production pipelines, cluster computing

        RAPIDS: GPU-accelerated processing using cuDF
            Best for: 100MB - 100GB with GPU available, high-performance analytics

        AUTO: Automatic selection based on data size and hardware
            Uses EngineRecommender to choose optimal engine

    Example:
        >>> from dataforge.core.base import EngineType
        >>>
        >>> # Explicit engine selection
        >>> df = DataFrame.read_csv("data.csv", engine=EngineType.SPARK)
        >>>
        >>> # Automatic selection
        >>> df = DataFrame.read_csv("data.csv", engine=EngineType.AUTO)
    """
    PANDAS = "pandas"
    POLARS = "polars"
    SPARK = "spark"
    RAPIDS = "rapids"
    AUTO = "auto"

    def __str__(self) -> str:
        return self.value


class FileFormat(str, Enum):
    """
    Supported file formats for read/write operations.

    Each format has different characteristics and use cases:

    Attributes:
        CSV: Plain text, human-readable, universally supported
            Pros: Universal compatibility
            Cons: Slow, no schema, large file size

        PARQUET: Columnar format, optimized for analytics
            Pros: Fast reads, compression, schema preservation
            Cons: Not human-readable

        JSON: Structured text format
            Pros: Human-readable, flexible schema
            Cons: Verbose, slow for large data

        ORC: Columnar format, optimized for Hive
            Pros: Great compression, predicate pushdown
            Cons: Less universal than Parquet

        DELTA: Delta Lake format (Parquet + transaction log)
            Pros: ACID transactions, time travel, schema evolution
            Cons: Requires Delta Lake library

        AVRO: Row-based binary format
            Pros: Schema evolution, compact
            Cons: Not optimized for analytics queries
    """
    CSV = "csv"
    PARQUET = "parquet"
    JSON = "json"
    ORC = "orc"
    DELTA = "delta"
    AVRO = "avro"


class JoinType(str, Enum):
    """
    Types of join operations supported across all engines.

    All engines implement these join types with consistent semantics.

    Attributes:
        INNER: Only matching rows from both DataFrames
        LEFT: All rows from left, matching from right (nulls for non-matches)
        RIGHT: All rows from right, matching from left
        OUTER: All rows from both DataFrames
        CROSS: Cartesian product (every row paired with every row)
        LEFT_SEMI: Rows from left that have match in right (no right columns)
        LEFT_ANTI: Rows from left that have NO match in right
    """
    INNER = "inner"
    LEFT = "left"
    RIGHT = "right"
    OUTER = "outer"
    CROSS = "cross"
    LEFT_SEMI = "left_semi"
    LEFT_ANTI = "left_anti"


@dataclass
class ReadOptions:
    """
    Configuration options for read operations.

    These options are normalized across all engines to provide consistent
    behavior regardless of the underlying implementation.

    Attributes:
        header: First row contains column names (CSV/JSON)
        infer_schema: Automatically infer column types
            Performance note: Requires extra pass over data for CSV

        schema: Explicit schema definition (engine-specific format)
            Recommended for production to avoid inference overhead

        partition_columns: Columns to use for partitioned reads
            Enables partition pruning for better performance

        columns: Specific columns to read (projection pushdown)
            Performance gain: Significant for wide tables

        filter: Filter expression to apply during read (predicate pushdown)
            Performance gain: Reduces I/O for partitioned/columnar data

        sample_fraction: Fraction of data to sample (0.0-1.0)
            Useful for exploratory analysis on large datasets

        encoding: Character encoding for text files
            Default: UTF-8

        delimiter: Field delimiter for CSV files
            Default: ","

        null_values: Strings to interpret as null values
            Default: ["", "NULL", "null", "NA", "N/A"]

        date_format: Format string for parsing dates
            Default: "yyyy-MM-dd"

        timestamp_format: Format string for parsing timestamps
            Default: "yyyy-MM-dd HH:mm:ss"

    Example:
        >>> options = ReadOptions(
        ...     columns=["id", "name", "amount"],
        ...     filter="amount > 1000",
        ...     infer_schema=False,
        ...     schema={"id": "int", "name": "string", "amount": "double"}
        ... )
        >>> df = engine.read_parquet("data.parquet", options)
    """
    header: bool = True
    infer_schema: bool = True
    schema: Optional[Dict[str, str]] = None
    partition_columns: Optional[List[str]] = None
    columns: Optional[List[str]] = None
    filter: Optional[str] = None
    sample_fraction: Optional[float] = None
    encoding: str = "utf-8"
    delimiter: str = ","
    null_values: List[str] = field(
        default_factory=lambda: ["", "NULL", "null", "NA", "N/A", "nan"]
    )
    date_format: str = "yyyy-MM-dd"
    timestamp_format: str = "yyyy-MM-dd HH:mm:ss"
    multiline: bool = False  # For JSON
    compression: Optional[str] = None  # gzip, snappy, etc.

    def __post_init__(self) -> None:
        """Validate options."""
        if self.sample_fraction is not None:
            if not 0.0 < self.sample_fraction <= 1.0:
                raise ValueError("sample_fraction must be between 0.0 and 1.0")


@dataclass
class WriteOptions:
    """
    Configuration options for write operations.

    These options control how data is persisted across all engines.

    Attributes:
        mode: Write mode - 'overwrite', 'append', 'ignore', 'error'
            overwrite: Replace existing data
            append: Add to existing data
            ignore: Do nothing if data exists
            error: Raise error if data exists (default)

        partition_by: Columns to partition the output by
            Creates directory structure: col1=val1/col2=val2/
            Performance impact: Critical for large datasets

        compression: Compression codec to use
            Options: 'snappy' (default for Parquet), 'gzip', 'lz4', 'zstd'
            Trade-off: CPU time vs file size vs read speed

        coalesce: Number of output files to coalesce to
            Use to reduce small file problem
            Warning: May cause OOM for large datasets

        max_records_per_file: Maximum rows per output file
            Helps control file sizes for partitioned writes

        header: Write header row (CSV)

        index: Write index column (Pandas-specific)

        optimize: Run OPTIMIZE after write (Delta Lake)
            Compacts small files for better read performance

        z_order_by: Columns for Z-ORDER optimization (Delta Lake)
            Improves query performance for filtered reads

    Example:
        >>> options = WriteOptions(
        ...     mode="overwrite",
        ...     partition_by=["year", "month"],
        ...     compression="snappy",
        ...     optimize=True,
        ...     z_order_by=["customer_id"]
        ... )
        >>> engine.write_parquet(df, "output/", options)
    """
    mode: str = "error"
    partition_by: Optional[List[str]] = None
    compression: Optional[str] = "snappy"
    coalesce: Optional[int] = None
    max_records_per_file: Optional[int] = None
    header: bool = True
    index: bool = False
    optimize: bool = False
    z_order_by: Optional[List[str]] = None
    merge_schema: bool = False  # Delta Lake schema evolution

    def __post_init__(self) -> None:
        """Validate options."""
        valid_modes = {"overwrite", "append", "ignore", "error", "errorifexists"}
        if self.mode.lower() not in valid_modes:
            raise ValueError(f"mode must be one of {valid_modes}")
        self.mode = self.mode.lower()


class DataFrameEngine(ABC, Generic[T]):
    """
    Abstract base class for all DataFrame engine implementations.

    This class defines the contract that Pandas, Spark, and RAPIDS engines
    must implement. It ensures consistent API across all backends while
    allowing engine-specific optimizations.

    Type Parameters:
        T: The native DataFrame type (pd.DataFrame, spark DataFrame, cudf.DataFrame)

    Implementation Requirements:
        - All abstract methods must be implemented
        - Methods should apply engine-specific best practices
        - Error handling should use DataForge exceptions
        - Performance-critical operations should be optimized

    Example Implementation:
        >>> class MyEngine(DataFrameEngine[pd.DataFrame]):
        ...     def read_csv(self, path: str, options: ReadOptions) -> pd.DataFrame:
        ...         return pd.read_csv(path, **self._convert_options(options))
    """

    @property
    @abstractmethod
    def engine_type(self) -> EngineType:
        """Return the engine type identifier."""
        pass

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the engine is available in the current environment."""
        pass

    # ==========================================================================
    # READ OPERATIONS
    # ==========================================================================

    @abstractmethod
    def read_csv(
        self,
        path: Union[str, Path, List[str]],
        options: Optional[ReadOptions] = None
    ) -> T:
        """
        Read CSV file(s) into a DataFrame.

        Args:
            path: Path to CSV file(s). Supports glob patterns.
            options: Read configuration options

        Returns:
            DataFrame with CSV data

        Raises:
            FileNotFoundError: If path does not exist
            DataForgeError: If read fails

        Performance Tips:
            - Provide explicit schema to avoid inference overhead
            - Use columns parameter for projection pushdown
            - Enable parallel reads for multiple files
        """
        pass

    @abstractmethod
    def read_parquet(
        self,
        path: Union[str, Path, List[str]],
        options: Optional[ReadOptions] = None
    ) -> T:
        """
        Read Parquet file(s) into a DataFrame.

        Parquet is recommended for analytical workloads due to:
            - Columnar storage for efficient column access
            - Built-in compression
            - Schema preservation
            - Predicate pushdown support

        Args:
            path: Path to Parquet file(s) or directory
            options: Read configuration options

        Returns:
            DataFrame with Parquet data

        Performance Tips:
            - Use filter option for predicate pushdown
            - Use columns option for projection pushdown
            - Partitioned datasets benefit from partition pruning
        """
        pass

    @abstractmethod
    def read_json(
        self,
        path: Union[str, Path, List[str]],
        options: Optional[ReadOptions] = None
    ) -> T:
        """
        Read JSON file(s) into a DataFrame.

        Args:
            path: Path to JSON file(s)
            options: Read configuration options

        Returns:
            DataFrame with JSON data

        Note:
            JSON is slower than Parquet/CSV for large datasets.
            Consider converting to Parquet for repeated reads.
        """
        pass

    @abstractmethod
    def read_delta(
        self,
        path: Union[str, Path],
        options: Optional[ReadOptions] = None,
        version: Optional[int] = None,
        timestamp: Optional[str] = None
    ) -> T:
        """
        Read Delta Lake table into a DataFrame.

        Delta Lake provides ACID transactions, time travel, and
        schema evolution on top of Parquet files.

        Args:
            path: Path to Delta table
            options: Read configuration options
            version: Specific version to read (time travel)
            timestamp: Timestamp for time travel query

        Returns:
            DataFrame with Delta table data

        Example:
            >>> # Read latest version
            >>> df = engine.read_delta("/path/to/delta_table")
            >>>
            >>> # Read specific version
            >>> df = engine.read_delta("/path/to/delta_table", version=5)
            >>>
            >>> # Read as of timestamp
            >>> df = engine.read_delta(
            ...     "/path/to/delta_table",
            ...     timestamp="2024-01-01 00:00:00"
            ... )
        """
        pass

    # ==========================================================================
    # WRITE OPERATIONS
    # ==========================================================================

    @abstractmethod
    def write_csv(
        self,
        df: T,
        path: Union[str, Path],
        options: Optional[WriteOptions] = None
    ) -> None:
        """
        Write DataFrame to CSV file(s).

        Args:
            df: DataFrame to write
            path: Output path
            options: Write configuration options

        Performance Tips:
            - Use coalesce to control output file count
            - Consider Parquet for better performance
        """
        pass

    @abstractmethod
    def write_parquet(
        self,
        df: T,
        path: Union[str, Path],
        options: Optional[WriteOptions] = None
    ) -> None:
        """
        Write DataFrame to Parquet file(s).

        Args:
            df: DataFrame to write
            path: Output path
            options: Write configuration options

        Performance Tips:
            - Use partition_by for large datasets
            - Use snappy compression (default) for balanced speed/size
            - Use zstd compression for maximum compression
        """
        pass

    @abstractmethod
    def write_delta(
        self,
        df: T,
        path: Union[str, Path],
        options: Optional[WriteOptions] = None
    ) -> None:
        """
        Write DataFrame to Delta Lake table.

        Args:
            df: DataFrame to write
            path: Output path for Delta table
            options: Write configuration options

        Delta Lake Benefits:
            - ACID transactions
            - Time travel for data versioning
            - Schema evolution support
            - OPTIMIZE and VACUUM for performance
        """
        pass

    # ==========================================================================
    # TRANSFORMATION OPERATIONS
    # ==========================================================================

    @abstractmethod
    def filter(self, df: T, condition: str) -> T:
        """
        Filter rows based on a condition.

        Args:
            df: Input DataFrame
            condition: SQL-like filter expression

        Returns:
            Filtered DataFrame

        Example:
            >>> filtered = engine.filter(df, "age > 18 AND status = 'active'")

        Performance Tips:
            - Filter early in the pipeline to reduce data volume
            - Use indexed columns when available
        """
        pass

    @abstractmethod
    def select(self, df: T, columns: List[str]) -> T:
        """
        Select specific columns from DataFrame.

        Args:
            df: Input DataFrame
            columns: List of column names to select

        Returns:
            DataFrame with selected columns

        Example:
            >>> result = engine.select(df, ["id", "name", "amount"])
        """
        pass

    @abstractmethod
    def rename(self, df: T, columns: Dict[str, str]) -> T:
        """
        Rename columns in DataFrame.

        Args:
            df: Input DataFrame
            columns: Mapping of old names to new names

        Returns:
            DataFrame with renamed columns

        Example:
            >>> result = engine.rename(df, {"old_name": "new_name"})
        """
        pass

    @abstractmethod
    def with_column(
        self,
        df: T,
        name: str,
        expression: Union[str, Callable]
    ) -> T:
        """
        Add or replace a column using an expression.

        Args:
            df: Input DataFrame
            name: Name for the new/replaced column
            expression: SQL expression or callable to compute values

        Returns:
            DataFrame with new column

        Example:
            >>> result = engine.with_column(df, "total", "price * quantity")
        """
        pass

    @abstractmethod
    def drop(self, df: T, columns: List[str]) -> T:
        """
        Drop columns from DataFrame.

        Args:
            df: Input DataFrame
            columns: List of column names to drop

        Returns:
            DataFrame without dropped columns
        """
        pass

    @abstractmethod
    def distinct(self, df: T) -> T:
        """
        Return distinct rows.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with duplicate rows removed

        Performance Note:
            Requires shuffle operation in distributed engines.
        """
        pass

    @abstractmethod
    def sort(
        self,
        df: T,
        columns: List[str],
        ascending: Union[bool, List[bool]] = True
    ) -> T:
        """
        Sort DataFrame by columns.

        Args:
            df: Input DataFrame
            columns: Columns to sort by
            ascending: Sort direction(s)

        Returns:
            Sorted DataFrame

        Performance Note:
            Expensive operation requiring full data shuffle in Spark.
        """
        pass

    @abstractmethod
    def limit(self, df: T, n: int) -> T:
        """
        Return first n rows.

        Args:
            df: Input DataFrame
            n: Number of rows to return

        Returns:
            DataFrame with at most n rows
        """
        pass

    # ==========================================================================
    # AGGREGATION OPERATIONS
    # ==========================================================================

    @abstractmethod
    def groupby(
        self,
        df: T,
        columns: List[str],
        aggregations: Dict[str, Union[str, List[str]]]
    ) -> T:
        """
        Group by columns and apply aggregations.

        Args:
            df: Input DataFrame
            columns: Columns to group by
            aggregations: Mapping of column to aggregation function(s)
                Supported: 'sum', 'avg', 'mean', 'min', 'max', 'count',
                          'first', 'last', 'std', 'var', 'collect_list'

        Returns:
            Aggregated DataFrame

        Example:
            >>> result = engine.groupby(
            ...     df,
            ...     columns=["region", "product"],
            ...     aggregations={
            ...         "amount": ["sum", "avg"],
            ...         "quantity": "sum",
            ...         "id": "count"
            ...     }
            ... )
        """
        pass

    @abstractmethod
    def agg(
        self,
        df: T,
        aggregations: Dict[str, Union[str, List[str]]]
    ) -> T:
        """
        Apply aggregations without grouping.

        Args:
            df: Input DataFrame
            aggregations: Mapping of column to aggregation function(s)

        Returns:
            Single-row DataFrame with aggregation results

        Example:
            >>> result = engine.agg(df, {
            ...     "amount": "sum",
            ...     "id": "count"
            ... })
        """
        pass

    # ==========================================================================
    # JOIN OPERATIONS
    # ==========================================================================

    @abstractmethod
    def join(
        self,
        left: T,
        right: T,
        on: Optional[Union[str, List[str]]] = None,
        left_on: Optional[Union[str, List[str]]] = None,
        right_on: Optional[Union[str, List[str]]] = None,
        how: JoinType = JoinType.INNER
    ) -> T:
        """
        Join two DataFrames.

        Args:
            left: Left DataFrame
            right: Right DataFrame
            on: Column(s) to join on (when same name in both)
            left_on: Column(s) from left DataFrame
            right_on: Column(s) from right DataFrame
            how: Type of join to perform

        Returns:
            Joined DataFrame

        Performance Tips:
            - Use broadcast joins for small right tables
            - Ensure join keys have same types
            - Consider pre-sorting for merge joins
            - Filter before joining to reduce data volume

        Example:
            >>> result = engine.join(
            ...     orders, customers,
            ...     left_on="customer_id",
            ...     right_on="id",
            ...     how=JoinType.LEFT
            ... )
        """
        pass

    # ==========================================================================
    # UTILITY OPERATIONS
    # ==========================================================================

    @abstractmethod
    def count(self, df: T) -> int:
        """
        Count number of rows.

        Args:
            df: Input DataFrame

        Returns:
            Number of rows
        """
        pass

    @abstractmethod
    def columns(self, df: T) -> List[str]:
        """
        Get column names.

        Args:
            df: Input DataFrame

        Returns:
            List of column names
        """
        pass

    @abstractmethod
    def dtypes(self, df: T) -> Dict[str, str]:
        """
        Get column data types.

        Args:
            df: Input DataFrame

        Returns:
            Mapping of column names to type names
        """
        pass

    @abstractmethod
    def schema(self, df: T) -> Dict[str, Any]:
        """
        Get full schema information.

        Args:
            df: Input DataFrame

        Returns:
            Schema representation (engine-specific format)
        """
        pass

    @abstractmethod
    def to_pandas(self, df: T) -> "pd.DataFrame":
        """
        Convert to pandas DataFrame.

        Args:
            df: Input DataFrame

        Returns:
            pandas DataFrame

        Warning:
            Collects all data to driver memory. Use with caution on large datasets.
        """
        pass

    @abstractmethod
    def from_pandas(self, df: "pd.DataFrame") -> T:
        """
        Create engine DataFrame from pandas DataFrame.

        Args:
            df: pandas DataFrame

        Returns:
            Engine-native DataFrame
        """
        pass

    @abstractmethod
    def cache(self, df: T) -> T:
        """
        Cache DataFrame for repeated use.

        Args:
            df: DataFrame to cache

        Returns:
            Cached DataFrame

        Note:
            Remember to uncache when done to free memory.
        """
        pass

    @abstractmethod
    def uncache(self, df: T) -> T:
        """
        Remove DataFrame from cache.

        Args:
            df: Cached DataFrame

        Returns:
            Uncached DataFrame
        """
        pass

    @abstractmethod
    def show(self, df: T, n: int = 20, truncate: bool = True) -> None:
        """
        Display DataFrame rows.

        Args:
            df: DataFrame to display
            n: Number of rows to show
            truncate: Truncate long strings
        """
        pass

    @abstractmethod
    def collect(self, df: T) -> List[Dict[str, Any]]:
        """
        Collect all rows as list of dictionaries.

        Args:
            df: DataFrame to collect

        Returns:
            List of row dictionaries

        Warning:
            Collects all data to driver memory.
        """
        pass

    @abstractmethod
    def head(self, df: T, n: int = 5) -> List[Dict[str, Any]]:
        """
        Get first n rows as list of dictionaries.

        Args:
            df: Input DataFrame
            n: Number of rows

        Returns:
            First n rows as dictionaries
        """
        pass
