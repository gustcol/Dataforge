"""
DataForge Databricks Optimizations

Databricks-specific performance optimizations and configurations.

Features:
    - Spark configuration optimization
    - Adaptive Query Execution (AQE) tuning
    - Shuffle optimization
    - Caching strategies
    - Photon recommendations

Best Practices:
    1. Enable AQE for better query performance
    2. Tune shuffle partitions based on data size
    3. Use broadcast joins for small tables
    4. Optimize file sizes (128MB-1GB target)
    5. Leverage Photon for vectorized execution

Example:
    >>> from dataforge.databricks import optimize_spark_config, get_photon_recommendations
    >>>
    >>> # Apply optimizations
    >>> optimize_spark_config(spark, data_size_gb=50)
    >>>
    >>> # Get Photon recommendations
    >>> recommendations = get_photon_recommendations(spark)
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TYPE_CHECKING
import logging

if TYPE_CHECKING:
    from pyspark.sql import SparkSession

logger = logging.getLogger(__name__)


@dataclass
class OptimizationConfig:
    """Configuration for Spark optimizations.

    Attributes:
        shuffle_partitions: Number of shuffle partitions
        broadcast_threshold_mb: Max size for broadcast joins (MB)
        adaptive_enabled: Enable AQE
        adaptive_coalesce: Enable partition coalescing
        adaptive_skew_join: Enable skew join optimization
        max_partition_bytes_mb: Target partition size (MB)
        min_partition_num: Minimum partitions after coalesce
        photon_enabled: Enable Photon engine
    """
    shuffle_partitions: int = 200
    broadcast_threshold_mb: int = 10
    adaptive_enabled: bool = True
    adaptive_coalesce: bool = True
    adaptive_skew_join: bool = True
    max_partition_bytes_mb: int = 128
    min_partition_num: int = 1
    photon_enabled: bool = True


def optimize_spark_config(
    spark: "SparkSession",
    data_size_gb: Optional[float] = None,
    config: Optional[OptimizationConfig] = None
) -> Dict[str, str]:
    """
    Apply optimized Spark configurations.

    Automatically tunes Spark settings based on data size and
    best practices for Databricks environments.

    Args:
        spark: SparkSession to configure
        data_size_gb: Estimated data size in GB (for auto-tuning)
        config: Optional custom configuration

    Returns:
        Dictionary of applied configurations

    Example:
        >>> # Auto-tune for 50GB dataset
        >>> applied = optimize_spark_config(spark, data_size_gb=50)
        >>> print(f"Applied {len(applied)} optimizations")
    """
    if config is None:
        config = OptimizationConfig()

    # Auto-tune based on data size
    if data_size_gb is not None:
        config = _auto_tune_config(config, data_size_gb)

    applied = {}

    # Adaptive Query Execution
    if config.adaptive_enabled:
        aqe_configs = {
            "spark.sql.adaptive.enabled": "true",
            "spark.sql.adaptive.coalescePartitions.enabled": str(config.adaptive_coalesce).lower(),
            "spark.sql.adaptive.skewJoin.enabled": str(config.adaptive_skew_join).lower(),
            "spark.sql.adaptive.advisoryPartitionSizeInBytes": f"{config.max_partition_bytes_mb}m",
            "spark.sql.adaptive.coalescePartitions.minPartitionNum": str(config.min_partition_num),
        }
        for key, value in aqe_configs.items():
            spark.conf.set(key, value)
            applied[key] = value
            logger.debug(f"Set {key}={value}")

    # Shuffle partitions
    spark.conf.set("spark.sql.shuffle.partitions", str(config.shuffle_partitions))
    applied["spark.sql.shuffle.partitions"] = str(config.shuffle_partitions)

    # Broadcast join threshold
    broadcast_bytes = config.broadcast_threshold_mb * 1024 * 1024
    spark.conf.set("spark.sql.autoBroadcastJoinThreshold", str(broadcast_bytes))
    applied["spark.sql.autoBroadcastJoinThreshold"] = str(broadcast_bytes)

    # File-based optimizations
    file_configs = {
        "spark.sql.files.maxPartitionBytes": f"{config.max_partition_bytes_mb}m",
        "spark.sql.files.openCostInBytes": "4194304",  # 4MB
    }
    for key, value in file_configs.items():
        spark.conf.set(key, value)
        applied[key] = value

    logger.info(f"Applied {len(applied)} Spark optimizations")
    return applied


def _auto_tune_config(config: OptimizationConfig, data_size_gb: float) -> OptimizationConfig:
    """Auto-tune configuration based on data size."""
    # Rule of thumb: 128MB per partition
    estimated_partitions = int(data_size_gb * 1024 / 128)

    # Clamp to reasonable range
    config.shuffle_partitions = max(
        config.min_partition_num,
        min(estimated_partitions, 10000)
    )

    # Increase broadcast threshold for larger datasets
    if data_size_gb > 100:
        config.broadcast_threshold_mb = 100
    elif data_size_gb > 10:
        config.broadcast_threshold_mb = 50

    # Adjust partition size for very large datasets
    if data_size_gb > 500:
        config.max_partition_bytes_mb = 256

    return config


def get_photon_recommendations(spark: "SparkSession") -> Dict[str, Any]:
    """
    Get Photon optimization recommendations.

    Analyzes current configuration and provides recommendations
    for maximizing Photon performance.

    Args:
        spark: SparkSession to analyze

    Returns:
        Dictionary with recommendations and current status

    Example:
        >>> recs = get_photon_recommendations(spark)
        >>> for rec in recs["recommendations"]:
        ...     print(f"- {rec}")
    """
    recommendations = []
    status = {}

    # Check Photon status
    try:
        photon_enabled = spark.conf.get("spark.databricks.photon.enabled", "false")
        status["photon_enabled"] = photon_enabled.lower() == "true"
    except Exception:
        status["photon_enabled"] = False

    if not status["photon_enabled"]:
        recommendations.append(
            "Enable Photon for 2-8x performance improvement on supported operations. "
            "Use Photon-enabled cluster or set spark.databricks.photon.enabled=true"
        )

    # Check AQE settings (important for Photon)
    try:
        aqe_enabled = spark.conf.get("spark.sql.adaptive.enabled", "true")
        status["aqe_enabled"] = aqe_enabled.lower() == "true"
    except Exception:
        status["aqe_enabled"] = True

    if not status["aqe_enabled"]:
        recommendations.append(
            "Enable Adaptive Query Execution (AQE) for better Photon performance: "
            "spark.sql.adaptive.enabled=true"
        )

    # Check file format recommendations
    recommendations.append(
        "Use Delta Lake format for best Photon performance. "
        "Delta provides data skipping and optimized I/O patterns."
    )

    # Check data types
    recommendations.append(
        "Prefer native types over complex types (arrays, maps, structs) "
        "for maximum Photon acceleration."
    )

    # Check UDF usage
    recommendations.append(
        "Avoid Python UDFs - use Spark SQL functions or Pandas UDFs (vectorized) "
        "for Photon compatibility."
    )

    return {
        "status": status,
        "recommendations": recommendations,
        "photon_supported_operations": [
            "Scan (Parquet, Delta, ORC)",
            "Filter",
            "Project",
            "Hash Aggregate",
            "Sort Merge Join",
            "Broadcast Hash Join",
            "Sort",
            "Window functions",
        ],
        "photon_unsupported": [
            "Python UDFs",
            "Complex type operations",
            "Some string functions",
            "User-defined aggregates",
        ],
    }


def configure_for_large_shuffle(spark: "SparkSession", data_size_gb: float) -> None:
    """
    Configure Spark for large shuffle operations.

    Applies configurations to handle large shuffle operations
    efficiently and avoid out-of-memory errors.

    Args:
        spark: SparkSession to configure
        data_size_gb: Estimated shuffle data size

    Example:
        >>> # Prepare for large join
        >>> configure_for_large_shuffle(spark, data_size_gb=100)
    """
    # Calculate partitions (aim for 128MB per partition)
    num_partitions = max(200, int(data_size_gb * 1024 / 128))

    configs = {
        "spark.sql.shuffle.partitions": str(num_partitions),
        "spark.sql.adaptive.enabled": "true",
        "spark.sql.adaptive.coalescePartitions.enabled": "true",
        "spark.sql.adaptive.skewJoin.enabled": "true",
        "spark.sql.adaptive.skewJoin.skewedPartitionFactor": "5",
        "spark.sql.adaptive.skewJoin.skewedPartitionThresholdInBytes": "256m",
    }

    for key, value in configs.items():
        spark.conf.set(key, value)

    logger.info(f"Configured for large shuffle: {num_partitions} partitions")


def configure_for_streaming(spark: "SparkSession") -> None:
    """
    Configure Spark for streaming workloads.

    Applies optimal configurations for Structured Streaming.

    Args:
        spark: SparkSession to configure
    """
    configs = {
        # Streaming-specific
        "spark.sql.streaming.schemaInference": "true",
        "spark.sql.streaming.checkpointLocation.autoCreate": "true",

        # Optimize for continuous processing
        "spark.sql.shuffle.partitions": "10",  # Lower for streaming
        "spark.sql.adaptive.enabled": "false",  # Disable AQE for streaming stability

        # State management
        "spark.sql.streaming.stateStore.providerClass":
            "com.databricks.sql.streaming.state.RocksDBStateStoreProvider",
    }

    for key, value in configs.items():
        try:
            spark.conf.set(key, value)
        except Exception as e:
            logger.debug(f"Could not set {key}: {e}")

    logger.info("Configured for streaming workloads")


def configure_for_ml(spark: "SparkSession") -> None:
    """
    Configure Spark for ML workloads.

    Applies optimal configurations for machine learning pipelines.

    Args:
        spark: SparkSession to configure
    """
    configs = {
        # Memory management for ML
        "spark.sql.execution.arrow.enabled": "true",
        "spark.sql.execution.arrow.pyspark.enabled": "true",
        "spark.sql.execution.arrow.maxRecordsPerBatch": "10000",

        # Optimize for iterative algorithms
        "spark.sql.adaptive.enabled": "true",
        "spark.locality.wait": "0s",  # Don't wait for data locality

        # Cache management
        "spark.sql.inMemoryColumnarStorage.compressed": "true",
        "spark.sql.inMemoryColumnarStorage.batchSize": "10000",
    }

    for key, value in configs.items():
        spark.conf.set(key, value)

    logger.info("Configured for ML workloads")


def get_cluster_recommendations(spark: "SparkSession") -> Dict[str, Any]:
    """
    Get cluster sizing recommendations.

    Analyzes current workload and provides recommendations
    for cluster sizing.

    Args:
        spark: SparkSession to analyze

    Returns:
        Dictionary with recommendations
    """
    recommendations = {}

    # Get current cluster info
    sc = spark.sparkContext
    num_executors = sc.getConf().get("spark.executor.instances", "unknown")
    executor_memory = sc.getConf().get("spark.executor.memory", "unknown")
    executor_cores = sc.getConf().get("spark.executor.cores", "unknown")

    recommendations["current_config"] = {
        "num_executors": num_executors,
        "executor_memory": executor_memory,
        "executor_cores": executor_cores,
    }

    recommendations["general_guidelines"] = [
        "Use 4-8 cores per executor for optimal parallelism",
        "Allocate 4-8 GB memory per core",
        "Use Delta Lake for best I/O performance",
        "Enable Photon for compute-heavy workloads",
        "Consider autoscaling for variable workloads",
    ]

    recommendations["workload_specific"] = {
        "etl": {
            "instance_type": "i3.xlarge or similar (storage optimized)",
            "cores_per_executor": 4,
            "memory_per_core_gb": 4,
            "notes": "Prioritize I/O throughput",
        },
        "ml": {
            "instance_type": "r5.xlarge or similar (memory optimized)",
            "cores_per_executor": 4,
            "memory_per_core_gb": 8,
            "notes": "Higher memory for model training",
        },
        "streaming": {
            "instance_type": "c5.xlarge or similar (compute optimized)",
            "cores_per_executor": 2,
            "memory_per_core_gb": 4,
            "notes": "Smaller executors for low latency",
        },
        "analytics": {
            "instance_type": "Photon-enabled cluster",
            "cores_per_executor": 8,
            "memory_per_core_gb": 4,
            "notes": "Leverage Photon for SQL workloads",
        },
    }

    return recommendations


def analyze_query_performance(spark: "SparkSession", query: str) -> Dict[str, Any]:
    """
    Analyze query and provide performance recommendations.

    Args:
        spark: SparkSession
        query: SQL query to analyze

    Returns:
        Dictionary with analysis and recommendations
    """
    analysis = {
        "query": query,
        "recommendations": [],
    }

    # Get query plan
    try:
        df = spark.sql(f"EXPLAIN EXTENDED {query}")
        plan = df.collect()[0][0]
        analysis["plan_summary"] = plan[:500] + "..." if len(plan) > 500 else plan
    except Exception as e:
        analysis["plan_error"] = str(e)
        return analysis

    # Check for common issues
    plan_lower = plan.lower()

    if "broadcasthashjoin" not in plan_lower and "join" in query.lower():
        analysis["recommendations"].append(
            "Consider using broadcast join for small tables: "
            "Use /*+ BROADCAST(small_table) */ hint"
        )

    if "filescan" in plan_lower and "partitionfilters" not in plan_lower:
        analysis["recommendations"].append(
            "Query may benefit from partition pruning. "
            "Add partition column filters to WHERE clause."
        )

    if "sort" in plan_lower:
        analysis["recommendations"].append(
            "Query includes sorting. Consider if ORDER BY is necessary, "
            "or use LIMIT to reduce sorted data."
        )

    if "exchange" in plan_lower:
        analysis["recommendations"].append(
            "Query involves shuffle (exchange). "
            "Consider repartitioning data or using broadcast joins."
        )

    return analysis


def enable_query_watchdog(
    spark: "SparkSession",
    max_runtime_minutes: int = 60,
    max_output_rows: int = 10_000_000
) -> None:
    """
    Enable query watchdog to prevent runaway queries.

    Args:
        spark: SparkSession
        max_runtime_minutes: Maximum query runtime
        max_output_rows: Maximum output rows
    """
    configs = {
        "spark.databricks.queryWatchdog.enabled": "true",
        "spark.databricks.queryWatchdog.maxQueryRuntimeMinutes": str(max_runtime_minutes),
        "spark.databricks.queryWatchdog.maxOutputRows": str(max_output_rows),
    }

    for key, value in configs.items():
        try:
            spark.conf.set(key, value)
        except Exception as e:
            logger.warning(f"Could not set {key}: {e}")

    logger.info(
        f"Query watchdog enabled: max runtime={max_runtime_minutes}min, "
        f"max rows={max_output_rows:,}"
    )
