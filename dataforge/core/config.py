"""
DataForge Configuration Management

This module provides configuration dataclasses for all aspects of the DataForge
framework including engine-specific settings, thresholds, and optimization parameters.

Configuration Classes:
    - DataForgeConfig: Main configuration container
    - EngineConfig: Engine selection thresholds
    - PandasConfig: Pandas-specific optimizations
    - SparkConfig: Spark-specific settings
    - RapidsConfig: RAPIDS/cuDF-specific settings

Example:
    >>> from dataforge.core.config import DataForgeConfig, SparkConfig
    >>>
    >>> # Create custom configuration
    >>> config = DataForgeConfig(
    ...     spark=SparkConfig(
    ...         shuffle_partitions=200,
    ...         adaptive_enabled=True
    ...     ),
    ...     auto_engine_selection=True
    ... )
    >>>
    >>> # Use with DataFrame
    >>> df = DataFrame.read_csv("data.csv", config=config)

Best Practices:
    - Use environment variables for sensitive configurations
    - Adjust shuffle_partitions based on cluster size (2-3x cores)
    - Enable AQE for Spark 3.0+ workloads
    - Set GPU memory fraction based on available VRAM
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import Enum


class EngineType(str, Enum):
    """
    Enumeration of supported processing engines.

    Attributes:
        PANDAS: In-memory processing with pandas
        POLARS: High-performance processing with Polars (Rust-backed)
        SPARK: Distributed processing with PySpark
        RAPIDS: GPU-accelerated processing with cuDF
        AUTO: Automatic engine selection based on data characteristics
    """
    PANDAS = "pandas"
    POLARS = "polars"
    SPARK = "spark"
    RAPIDS = "rapids"
    AUTO = "auto"

    def __str__(self) -> str:
        return self.value


@dataclass
class PandasConfig:
    """
    Configuration for the Pandas engine.

    This configuration controls Pandas-specific optimizations including
    memory management, dtype optimization, and query execution.

    Attributes:
        enable_copy_on_write: Enable copy-on-write mode (pandas 2.0+)
            Reduces memory usage by avoiding unnecessary copies.
            Performance gain: 10-50% memory reduction.

        optimize_dtypes: Automatically downcast numeric types
            Converts int64 to int32/int16/int8 where possible.
            Performance gain: 25-75% memory reduction.

        use_nullable_dtypes: Use nullable integer/boolean types
            Enables proper NA handling for integer columns.

        chunk_size: Default chunk size for reading large files
            Files larger than max_memory_mb are read in chunks.
            Recommended: 100,000 - 1,000,000 rows.

        max_memory_mb: Maximum memory to use before switching to chunked mode
            Default: 80% of available system memory.

        use_numexpr: Enable numexpr for faster query evaluation
            Performance gain: 2-10x for complex boolean expressions.

        categorical_threshold: Max unique values for automatic categorization
            Columns with fewer unique values become categorical dtype.
            Performance gain: 50-90% memory reduction for string columns.

    Example:
        >>> config = PandasConfig(
        ...     enable_copy_on_write=True,
        ...     optimize_dtypes=True,
        ...     chunk_size=500_000
        ... )
    """
    enable_copy_on_write: bool = True
    optimize_dtypes: bool = True
    use_nullable_dtypes: bool = True
    chunk_size: int = 100_000
    max_memory_mb: int = 4096
    use_numexpr: bool = True
    categorical_threshold: int = 50

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.chunk_size < 1000:
            raise ValueError("chunk_size must be at least 1000")
        if self.max_memory_mb < 256:
            raise ValueError("max_memory_mb must be at least 256")
        if self.categorical_threshold < 2:
            raise ValueError("categorical_threshold must be at least 2")


@dataclass
class PolarsConfig:
    """
    Configuration for the Polars engine.

    Polars is a high-performance DataFrame library written in Rust that
    provides lazy evaluation, parallel execution, and memory-efficient
    processing for medium-to-large datasets on a single node.

    Attributes:
        use_lazy: Enable lazy evaluation mode for query optimization
            Polars will build an execution plan and optimize it before running.
            Performance gain: Significant for complex query chains.

        streaming: Enable streaming mode for out-of-core processing
            Allows processing datasets larger than available memory.

        n_rows: Default number of rows for head/sample operations

        rechunk: Rechunk DataFrames after operations for contiguous memory
            Improves performance for subsequent operations at memory cost.

        parallel: Enable parallel execution of operations
            Uses all available CPU cores automatically.

        max_threads: Maximum number of threads for parallel operations
            Default: None (uses all available cores).

    Example:
        >>> config = PolarsConfig(
        ...     use_lazy=True,
        ...     streaming=True,
        ...     max_threads=8
        ... )
    """
    use_lazy: bool = True
    streaming: bool = False
    n_rows: int = 5
    rechunk: bool = True
    parallel: bool = True
    max_threads: Optional[int] = None

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.n_rows < 1:
            raise ValueError("n_rows must be at least 1")
        if self.max_threads is not None and self.max_threads < 1:
            raise ValueError("max_threads must be at least 1")


@dataclass
class SparkConfig:
    """
    Configuration for the Spark engine.

    This configuration controls Spark-specific optimizations including
    Adaptive Query Execution (AQE), shuffle settings, and Delta Lake integration.

    Attributes:
        shuffle_partitions: Number of partitions for shuffle operations
            Default: 200. Recommended: 2-3x cluster cores.
            Performance impact: Critical for joins and aggregations.

        adaptive_enabled: Enable Adaptive Query Execution (Spark 3.0+)
            Dynamically adjusts partitions and join strategies.
            Performance gain: 10-50% for most workloads.

        adaptive_coalesce_partitions: Auto-coalesce small partitions
            Reduces overhead from many small tasks.
            Performance gain: Significant for skewed data.

        broadcast_threshold_mb: Max size for broadcast joins (MB)
            Tables smaller than this are broadcast to all executors.
            Default: 10MB. Max recommended: 100MB.
            Performance gain: 10-100x for small dimension tables.

        auto_broadcast_join: Automatically broadcast small tables
            Spark analyzes table sizes and broadcasts when beneficial.

        delta_optimized_writes: Enable Delta Lake optimized writes
            Reduces small file problem and improves read performance.
            Performance gain: 2-5x read performance improvement.

        delta_auto_compact: Automatically compact Delta files
            Runs OPTIMIZE automatically based on file count.

        photon_enabled: Enable Databricks Photon runtime
            Vectorized query engine for 2-8x performance improvement.
            Requires Databricks runtime with Photon support.

        cache_storage_level: Default storage level for caching
            Options: MEMORY_ONLY, MEMORY_AND_DISK, DISK_ONLY
            Default: MEMORY_AND_DISK for reliability.

    Example:
        >>> config = SparkConfig(
        ...     shuffle_partitions=400,  # For large cluster
        ...     adaptive_enabled=True,
        ...     broadcast_threshold_mb=50,
        ...     photon_enabled=True  # Databricks only
        ... )
    """
    shuffle_partitions: int = 200
    adaptive_enabled: bool = True
    adaptive_coalesce_partitions: bool = True
    adaptive_skew_join: bool = True
    broadcast_threshold_mb: int = 10
    auto_broadcast_join: bool = True
    delta_optimized_writes: bool = True
    delta_auto_compact: bool = False
    photon_enabled: bool = False
    cache_storage_level: str = "MEMORY_AND_DISK"
    executor_memory: Optional[str] = None
    driver_memory: Optional[str] = None
    extra_configs: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.shuffle_partitions < 1:
            raise ValueError("shuffle_partitions must be positive")
        if self.broadcast_threshold_mb < 0:
            raise ValueError("broadcast_threshold_mb cannot be negative")
        valid_storage_levels = {
            "MEMORY_ONLY", "MEMORY_AND_DISK", "DISK_ONLY",
            "MEMORY_ONLY_SER", "MEMORY_AND_DISK_SER"
        }
        if self.cache_storage_level not in valid_storage_levels:
            raise ValueError(f"cache_storage_level must be one of {valid_storage_levels}")

    def to_spark_conf(self) -> Dict[str, str]:
        """
        Convert configuration to Spark configuration dictionary.

        Returns:
            Dictionary of Spark configuration key-value pairs

        Example:
            >>> conf = config.to_spark_conf()
            >>> spark = SparkSession.builder.config(conf=conf).getOrCreate()
        """
        conf = {
            "spark.sql.shuffle.partitions": str(self.shuffle_partitions),
            "spark.sql.adaptive.enabled": str(self.adaptive_enabled).lower(),
            "spark.sql.adaptive.coalescePartitions.enabled": str(
                self.adaptive_coalesce_partitions
            ).lower(),
            "spark.sql.adaptive.skewJoin.enabled": str(
                self.adaptive_skew_join
            ).lower(),
            "spark.sql.autoBroadcastJoinThreshold": str(
                self.broadcast_threshold_mb * 1024 * 1024
            ),
        }

        if self.delta_optimized_writes:
            conf["spark.databricks.delta.optimizeWrite.enabled"] = "true"

        if self.delta_auto_compact:
            conf["spark.databricks.delta.autoCompact.enabled"] = "true"

        if self.executor_memory:
            conf["spark.executor.memory"] = self.executor_memory

        if self.driver_memory:
            conf["spark.driver.memory"] = self.driver_memory

        conf.update(self.extra_configs)

        return conf


@dataclass
class RapidsConfig:
    """
    Configuration for the RAPIDS/cuDF engine.

    This configuration controls GPU memory management, fallback behavior,
    and RAPIDS-specific optimizations.

    Attributes:
        gpu_memory_fraction: Fraction of GPU memory to allocate (0.0-1.0)
            Default: 0.8 (80% of available VRAM).
            Leave headroom for system and other processes.

        enable_spilling: Enable GPU memory spilling to CPU
            Allows processing larger-than-VRAM datasets.
            Performance impact: Significant when spilling occurs.

        pool_allocator: Use memory pool for faster allocations
            Options: 'default', 'managed', 'pool'
            Performance gain: 2-5x for many small allocations.

        fallback_to_pandas: Auto-fallback to pandas when GPU unavailable
            Enables graceful degradation without code changes.

        chunk_size_mb: Chunk size for GPU processing (MB)
            Split large datasets into GPU-manageable chunks.
            Default: GPU memory * gpu_memory_fraction / 2.

        enable_strings_udf: Enable string UDF acceleration
            GPU-accelerated string operations.
            Performance gain: 10-100x for string operations.

        nvtx_annotations: Enable NVTX for profiling
            Useful for debugging with Nsight Systems.

    Example:
        >>> config = RapidsConfig(
        ...     gpu_memory_fraction=0.7,
        ...     enable_spilling=True,
        ...     fallback_to_pandas=True
        ... )

    Hardware Requirements:
        - NVIDIA GPU with CUDA support (compute capability 6.0+)
        - CUDA 11.x or 12.x runtime
        - Sufficient GPU memory (recommended: 16GB+)
    """
    gpu_memory_fraction: float = 0.8
    enable_spilling: bool = True
    pool_allocator: str = "default"
    fallback_to_pandas: bool = True
    chunk_size_mb: int = 4096
    enable_strings_udf: bool = True
    nvtx_annotations: bool = False
    device_id: int = 0

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if not 0.0 < self.gpu_memory_fraction <= 1.0:
            raise ValueError("gpu_memory_fraction must be between 0.0 and 1.0")
        if self.chunk_size_mb < 64:
            raise ValueError("chunk_size_mb must be at least 64")
        valid_allocators = {"default", "managed", "pool", "arena"}
        if self.pool_allocator not in valid_allocators:
            raise ValueError(f"pool_allocator must be one of {valid_allocators}")


@dataclass
class EngineConfig:
    """
    Configuration for automatic engine selection.

    This configuration defines thresholds and rules for the engine
    recommendation system to select the optimal processing engine.

    Attributes:
        pandas_max_size_mb: Maximum data size for Pandas (MB)
            Above this threshold, Spark or RAPIDS is recommended.
            Default: 1024MB (1GB).

        rapids_min_size_mb: Minimum data size to benefit from RAPIDS (MB)
            GPU overhead makes RAPIDS slower for tiny datasets.
            Default: 100MB.

        rapids_max_size_mb: Maximum data size for single-GPU RAPIDS (MB)
            Based on GPU memory. Above this, use Spark.
            Default: 32GB (assumes 16GB GPU with spilling).

        spark_min_size_mb: Minimum size to justify Spark overhead (MB)
            Spark has initialization overhead not worth for small data.
            Default: 500MB.

        prefer_gpu: Prefer RAPIDS when GPU available and size appropriate
            Set False to prefer Spark for distributed processing.

        cluster_available: Is a Spark cluster available?
            Affects recommendation between local Pandas and Spark.

    Example:
        >>> config = EngineConfig(
        ...     pandas_max_size_mb=2048,  # Allow larger Pandas operations
        ...     prefer_gpu=True,
        ...     cluster_available=True
        ... )
    """
    pandas_max_size_mb: int = 1024
    rapids_min_size_mb: int = 100
    rapids_max_size_mb: int = 32768
    spark_min_size_mb: int = 500
    prefer_gpu: bool = True
    cluster_available: bool = False
    gpu_available: Optional[bool] = None  # Auto-detect if None

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.pandas_max_size_mb < 0:
            raise ValueError("pandas_max_size_mb cannot be negative")
        if self.rapids_min_size_mb > self.rapids_max_size_mb:
            raise ValueError("rapids_min_size_mb cannot exceed rapids_max_size_mb")


@dataclass
class DataForgeConfig:
    """
    Main configuration container for DataForge framework.

    This is the primary configuration class that aggregates all
    engine-specific configurations and global settings.

    Attributes:
        pandas: Pandas engine configuration
        spark: Spark engine configuration
        rapids: RAPIDS engine configuration
        engine_selection: Engine selection thresholds
        auto_engine_selection: Enable automatic engine selection
        default_engine: Default engine when auto-selection is disabled
        log_level: Logging verbosity level
        enable_benchmarking: Enable performance tracking
        databricks_mode: Running in Databricks environment

    Example:
        >>> from dataforge import DataForgeConfig
        >>>
        >>> config = DataForgeConfig(
        ...     auto_engine_selection=True,
        ...     spark=SparkConfig(shuffle_partitions=400),
        ...     enable_benchmarking=True
        ... )
        >>>
        >>> # Use globally
        >>> DataForgeConfig.set_global(config)
        >>>
        >>> # Or per-operation
        >>> df = DataFrame.read_csv("data.csv", config=config)
    """
    pandas: PandasConfig = field(default_factory=PandasConfig)
    polars: PolarsConfig = field(default_factory=PolarsConfig)
    spark: SparkConfig = field(default_factory=SparkConfig)
    rapids: RapidsConfig = field(default_factory=RapidsConfig)
    engine_selection: EngineConfig = field(default_factory=EngineConfig)
    auto_engine_selection: bool = True
    default_engine: EngineType = EngineType.PANDAS
    log_level: str = "INFO"
    enable_benchmarking: bool = False
    databricks_mode: bool = False

    # Global configuration singleton
    _global_config: Optional["DataForgeConfig"] = None

    def __post_init__(self) -> None:
        """Validate and normalize configuration."""
        valid_log_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if self.log_level.upper() not in valid_log_levels:
            raise ValueError(f"log_level must be one of {valid_log_levels}")
        self.log_level = self.log_level.upper()

        if isinstance(self.default_engine, str):
            self.default_engine = EngineType(self.default_engine)

    @classmethod
    def set_global(cls, config: "DataForgeConfig") -> None:
        """
        Set global configuration for all DataForge operations.

        Args:
            config: Configuration to use globally

        Example:
            >>> DataForgeConfig.set_global(DataForgeConfig(
            ...     auto_engine_selection=True,
            ...     enable_benchmarking=True
            ... ))
        """
        cls._global_config = config

    @classmethod
    def get_global(cls) -> "DataForgeConfig":
        """
        Get global configuration, creating default if not set.

        Returns:
            Global DataForgeConfig instance

        Example:
            >>> config = DataForgeConfig.get_global()
            >>> print(config.default_engine)
        """
        if cls._global_config is None:
            cls._global_config = cls()
        return cls._global_config

    @classmethod
    def reset_global(cls) -> None:
        """Reset global configuration to default."""
        cls._global_config = None

    def merge_with(self, other: Optional["DataForgeConfig"]) -> "DataForgeConfig":
        """
        Merge this configuration with another, preferring other's values.

        Args:
            other: Configuration to merge with (takes precedence)

        Returns:
            New merged configuration
        """
        if other is None:
            return self

        # Simple merge - other takes precedence for non-None values
        return DataForgeConfig(
            pandas=other.pandas if other.pandas else self.pandas,
            polars=other.polars if other.polars else self.polars,
            spark=other.spark if other.spark else self.spark,
            rapids=other.rapids if other.rapids else self.rapids,
            engine_selection=other.engine_selection or self.engine_selection,
            auto_engine_selection=other.auto_engine_selection,
            default_engine=other.default_engine,
            log_level=other.log_level or self.log_level,
            enable_benchmarking=other.enable_benchmarking,
            databricks_mode=other.databricks_mode,
        )
