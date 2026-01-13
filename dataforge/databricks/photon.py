"""
DataForge Photon Runtime Utilities

Databricks Photon is a vectorized query engine that provides
significant performance improvements for SQL and DataFrame workloads.

Features:
    - Photon compatibility checking
    - Performance analysis
    - Configuration recommendations
    - Fallback detection

Performance Characteristics:
    - 2-8x faster than standard Spark for supported operations
    - Native C++ execution engine
    - Vectorized processing
    - Optimized for Delta Lake

Supported Operations:
    - Scan (Parquet, Delta, ORC)
    - Filter and Project
    - Hash and Sort Merge Joins
    - Aggregations
    - Window functions
    - Sort operations

Not Supported:
    - Python/Scala UDFs (use Pandas UDFs instead)
    - Complex nested types
    - Some string operations
    - User-defined aggregates

Example:
    >>> from dataforge.databricks import PhotonAnalyzer
    >>>
    >>> analyzer = PhotonAnalyzer(spark)
    >>>
    >>> # Check if Photon is available
    >>> if analyzer.is_photon_enabled:
    ...     print("Photon is enabled!")
    >>>
    >>> # Analyze query compatibility
    >>> report = analyzer.analyze_query("SELECT * FROM sales WHERE amount > 100")
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING
import logging
import re

if TYPE_CHECKING:
    from pyspark.sql import SparkSession, DataFrame

logger = logging.getLogger(__name__)


@dataclass
class PhotonCompatibility:
    """Photon compatibility analysis result.

    Attributes:
        is_compatible: Whether operation is fully Photon-compatible
        supported_operations: Operations that will use Photon
        unsupported_operations: Operations that will fall back to Spark
        recommendations: Suggestions for improving compatibility
        estimated_speedup: Estimated performance improvement
    """
    is_compatible: bool = True
    supported_operations: Optional[List[str]] = None
    unsupported_operations: Optional[List[str]] = None
    recommendations: Optional[List[str]] = None
    estimated_speedup: str = "1-2x"

    def __post_init__(self):
        self.supported_operations = self.supported_operations or []
        self.unsupported_operations = self.unsupported_operations or []
        self.recommendations = self.recommendations or []


# Operations supported by Photon
PHOTON_SUPPORTED_OPERATIONS: Set[str] = {
    "scan",
    "filter",
    "project",
    "hashaggregate",
    "sortaggregate",
    "sortmergejoin",
    "broadcasthashjoin",
    "broadcastnestedloopjoin",
    "sort",
    "window",
    "exchange",
    "union",
    "expand",
    "generate",
    "coalesce",
}

# Operations that cause Photon fallback
PHOTON_UNSUPPORTED_PATTERNS: Dict[str, str] = {
    r"pythonudf": "Python UDF detected - use Spark SQL functions or Pandas UDFs",
    r"scalaudf": "Scala UDF detected - consider using built-in functions",
    r"mapelements": "Map transformation detected - use DataFrame operations",
    r"flatmapgroups": "FlatMapGroups detected - use window functions if possible",
    r"deserialize": "Deserialization detected - avoid RDD operations",
}


class PhotonAnalyzer:
    """
    Analyzer for Photon compatibility and performance.

    Provides utilities for checking Photon status, analyzing
    query compatibility, and optimizing for Photon execution.

    Example:
        >>> analyzer = PhotonAnalyzer(spark)
        >>>
        >>> # Check status
        >>> print(f"Photon enabled: {analyzer.is_photon_enabled}")
        >>>
        >>> # Analyze DataFrame operations
        >>> report = analyzer.analyze_dataframe(df)
        >>> if not report.is_compatible:
        ...     for rec in report.recommendations:
        ...         print(f"Recommendation: {rec}")
    """

    def __init__(self, spark: "SparkSession") -> None:
        """
        Initialize Photon analyzer.

        Args:
            spark: SparkSession to analyze
        """
        self.spark = spark

    @property
    def is_photon_enabled(self) -> bool:
        """Check if Photon is enabled on the cluster."""
        try:
            # Check cluster configuration
            photon_setting = self.spark.conf.get(
                "spark.databricks.photon.enabled", "false"
            )
            return photon_setting.lower() == "true"
        except Exception:
            return False

    @property
    def is_photon_available(self) -> bool:
        """Check if Photon is available (even if not enabled)."""
        try:
            # Check if running on Photon-capable cluster
            runtime = self.spark.conf.get(
                "spark.databricks.clusterUsageTags.sparkVersion", ""
            )
            return "photon" in runtime.lower()
        except Exception:
            return False

    def get_photon_status(self) -> Dict[str, Any]:
        """
        Get detailed Photon status.

        Returns:
            Dictionary with Photon configuration details
        """
        status = {
            "enabled": self.is_photon_enabled,
            "available": self.is_photon_available,
            "configurations": {},
        }

        # Get relevant configurations
        photon_configs = [
            "spark.databricks.photon.enabled",
            "spark.databricks.photon.parquetWriter.enabled",
            "spark.databricks.photon.window.enabled",
            "spark.databricks.photon.sort.enabled",
        ]

        for config in photon_configs:
            try:
                status["configurations"][config] = self.spark.conf.get(config)
            except Exception:
                status["configurations"][config] = "not set"

        return status

    def enable_photon(self) -> bool:
        """
        Enable Photon if available.

        Returns:
            True if Photon was enabled, False otherwise
        """
        if not self.is_photon_available:
            logger.warning("Photon is not available on this cluster")
            return False

        try:
            self.spark.conf.set("spark.databricks.photon.enabled", "true")
            logger.info("Photon enabled successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to enable Photon: {e}")
            return False

    def analyze_query(self, sql_query: str) -> PhotonCompatibility:
        """
        Analyze SQL query for Photon compatibility.

        Args:
            sql_query: SQL query to analyze

        Returns:
            PhotonCompatibility report

        Example:
            >>> report = analyzer.analyze_query('''
            ...     SELECT customer_id, SUM(amount)
            ...     FROM orders
            ...     WHERE date > '2024-01-01'
            ...     GROUP BY customer_id
            ... ''')
            >>> print(f"Compatible: {report.is_compatible}")
        """
        compatibility = PhotonCompatibility()

        try:
            # Get query plan
            explained = self.spark.sql(f"EXPLAIN EXTENDED {sql_query}")
            plan = explained.collect()[0][0].lower()

            # Analyze plan
            compatibility = self._analyze_plan(plan)

        except Exception as e:
            logger.error(f"Failed to analyze query: {e}")
            compatibility.is_compatible = False
            compatibility.recommendations.append(f"Query analysis failed: {e}")

        return compatibility

    def analyze_dataframe(self, df: "DataFrame") -> PhotonCompatibility:
        """
        Analyze DataFrame operations for Photon compatibility.

        Args:
            df: DataFrame to analyze

        Returns:
            PhotonCompatibility report
        """
        compatibility = PhotonCompatibility()

        try:
            # Get execution plan
            plan = df._jdf.queryExecution().executedPlan().toString().lower()

            # Analyze plan
            compatibility = self._analyze_plan(plan)

        except Exception as e:
            logger.error(f"Failed to analyze DataFrame: {e}")
            compatibility.is_compatible = False
            compatibility.recommendations.append(f"DataFrame analysis failed: {e}")

        return compatibility

    def _analyze_plan(self, plan: str) -> PhotonCompatibility:
        """Analyze query plan for Photon compatibility."""
        compatibility = PhotonCompatibility()
        supported = []
        unsupported = []

        # Check for supported operations
        for op in PHOTON_SUPPORTED_OPERATIONS:
            if op in plan:
                supported.append(op)

        # Check for unsupported patterns
        for pattern, message in PHOTON_UNSUPPORTED_PATTERNS.items():
            if re.search(pattern, plan):
                unsupported.append(pattern)
                compatibility.recommendations.append(message)
                compatibility.is_compatible = False

        compatibility.supported_operations = supported
        compatibility.unsupported_operations = unsupported

        # Estimate speedup
        if compatibility.is_compatible:
            if len(supported) > 5:
                compatibility.estimated_speedup = "4-8x"
            elif len(supported) > 2:
                compatibility.estimated_speedup = "2-4x"
            else:
                compatibility.estimated_speedup = "1-2x"
        else:
            compatibility.estimated_speedup = "Limited (partial fallback)"

        return compatibility

    def get_optimization_recommendations(self) -> List[str]:
        """
        Get recommendations for maximizing Photon performance.

        Returns:
            List of optimization recommendations
        """
        recommendations = []

        # Check Photon status
        if not self.is_photon_enabled:
            recommendations.append(
                "Enable Photon: Set spark.databricks.photon.enabled=true "
                "or use a Photon-enabled cluster"
            )

        # General recommendations
        recommendations.extend([
            "Use Delta Lake format for tables - provides best Photon integration "
            "with data skipping and optimized I/O",

            "Avoid Python UDFs - replace with Spark SQL functions. "
            "If UDFs are required, use Pandas UDFs (vectorized) instead",

            "Use native data types - avoid complex nested types "
            "(arrays, maps, structs) when possible",

            "Enable Adaptive Query Execution (AQE) - works well with Photon "
            "for dynamic optimization",

            "Keep Delta tables optimized - run OPTIMIZE regularly "
            "to maintain optimal file sizes (128MB-1GB)",

            "Use column pruning - SELECT only needed columns "
            "to reduce I/O and maximize Photon benefits",

            "Apply predicate pushdown - add filters early in queries "
            "to leverage Photon's optimized filtering",

            "Use broadcast joins for small tables (<10MB) - "
            "reduces shuffle and improves Photon performance",
        ])

        return recommendations

    def benchmark_query(
        self,
        sql_query: str,
        iterations: int = 3
    ) -> Dict[str, Any]:
        """
        Benchmark query with and without Photon.

        Note: Requires ability to toggle Photon setting.

        Args:
            sql_query: Query to benchmark
            iterations: Number of iterations per test

        Returns:
            Benchmark results
        """
        import time

        results = {
            "query": sql_query,
            "iterations": iterations,
            "photon_times": [],
            "spark_times": [],
        }

        # Clear cache
        self.spark.catalog.clearCache()

        # Benchmark with Photon (if enabled)
        if self.is_photon_enabled:
            for _ in range(iterations):
                start = time.time()
                self.spark.sql(sql_query).collect()
                results["photon_times"].append(time.time() - start)

        # Try to disable Photon and benchmark
        try:
            original_setting = self.spark.conf.get(
                "spark.databricks.photon.enabled", "true"
            )
            self.spark.conf.set("spark.databricks.photon.enabled", "false")

            self.spark.catalog.clearCache()

            for _ in range(iterations):
                start = time.time()
                self.spark.sql(sql_query).collect()
                results["spark_times"].append(time.time() - start)

            # Restore original setting
            self.spark.conf.set("spark.databricks.photon.enabled", original_setting)

        except Exception as e:
            results["benchmark_note"] = f"Could not toggle Photon: {e}"

        # Calculate statistics
        if results["photon_times"]:
            results["photon_avg"] = sum(results["photon_times"]) / len(results["photon_times"])

        if results["spark_times"]:
            results["spark_avg"] = sum(results["spark_times"]) / len(results["spark_times"])

        if results.get("photon_avg") and results.get("spark_avg"):
            results["speedup"] = results["spark_avg"] / results["photon_avg"]

        return results


def check_photon_compatibility(spark: "SparkSession", df: "DataFrame") -> Dict[str, Any]:
    """
    Quick check for Photon compatibility.

    Convenience function for checking DataFrame compatibility.

    Args:
        spark: SparkSession
        df: DataFrame to check

    Returns:
        Compatibility summary

    Example:
        >>> result = check_photon_compatibility(spark, my_df)
        >>> if result["compatible"]:
        ...     print(f"Expected speedup: {result['speedup']}")
    """
    analyzer = PhotonAnalyzer(spark)
    report = analyzer.analyze_dataframe(df)

    return {
        "compatible": report.is_compatible,
        "speedup": report.estimated_speedup,
        "supported_ops": report.supported_operations,
        "unsupported_ops": report.unsupported_operations,
        "recommendations": report.recommendations,
    }


def configure_photon_optimal(spark: "SparkSession") -> Dict[str, str]:
    """
    Apply optimal Photon configuration.

    Args:
        spark: SparkSession to configure

    Returns:
        Applied configurations

    Example:
        >>> configs = configure_photon_optimal(spark)
        >>> print(f"Applied {len(configs)} configurations")
    """
    configs = {
        # Core Photon settings
        "spark.databricks.photon.enabled": "true",
        "spark.databricks.photon.parquetWriter.enabled": "true",
        "spark.databricks.photon.window.enabled": "true",
        "spark.databricks.photon.sort.enabled": "true",

        # AQE for Photon
        "spark.sql.adaptive.enabled": "true",
        "spark.sql.adaptive.coalescePartitions.enabled": "true",
        "spark.sql.adaptive.skewJoin.enabled": "true",

        # I/O optimization
        "spark.sql.parquet.enableVectorizedReader": "true",
        "spark.sql.inMemoryColumnarStorage.compressed": "true",

        # Delta optimizations
        "spark.databricks.delta.optimizeWrite.enabled": "true",
        "spark.databricks.delta.autoCompact.enabled": "true",
    }

    applied = {}
    for key, value in configs.items():
        try:
            spark.conf.set(key, value)
            applied[key] = value
        except Exception as e:
            logger.debug(f"Could not set {key}: {e}")

    logger.info(f"Applied {len(applied)} Photon-optimal configurations")
    return applied
