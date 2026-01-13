"""
DataForge Databricks Context Manager

Provides utilities for detecting and managing the Databricks runtime environment.

Features:
    - Runtime detection (local vs Databricks)
    - Cluster information retrieval
    - Spark session management
    - DBUtils access
    - Secret scope integration

Best Practices:
    1. Use context detection for environment-specific code paths
    2. Always use secret scopes for credentials
    3. Leverage cluster tags for resource tracking
    4. Check runtime version for feature compatibility

Example:
    >>> from dataforge.databricks import DatabricksContext
    >>>
    >>> ctx = DatabricksContext()
    >>>
    >>> if ctx.is_databricks:
    ...     print(f"Running on Databricks {ctx.runtime_version}")
    ...     secret = ctx.get_secret("my-scope", "api-key")
    ... else:
    ...     print("Running locally")
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TYPE_CHECKING
import os
import logging

if TYPE_CHECKING:
    from pyspark.sql import SparkSession

logger = logging.getLogger(__name__)


@dataclass
class ClusterInfo:
    """Information about the Databricks cluster.

    Attributes:
        cluster_id: Unique cluster identifier
        cluster_name: Human-readable cluster name
        spark_version: Databricks Runtime version
        node_type: Instance type for worker nodes
        driver_node_type: Instance type for driver node
        num_workers: Number of worker nodes
        autoscale_min: Minimum workers (if autoscaling)
        autoscale_max: Maximum workers (if autoscaling)
        tags: Custom cluster tags
    """
    cluster_id: Optional[str] = None
    cluster_name: Optional[str] = None
    spark_version: Optional[str] = None
    node_type: Optional[str] = None
    driver_node_type: Optional[str] = None
    num_workers: Optional[int] = None
    autoscale_min: Optional[int] = None
    autoscale_max: Optional[int] = None
    tags: Optional[Dict[str, str]] = None


@dataclass
class RuntimeInfo:
    """Information about the Databricks Runtime.

    Attributes:
        version: Full runtime version string
        major_version: Major version number
        minor_version: Minor version number
        is_ml: Whether ML runtime is enabled
        is_photon: Whether Photon is enabled
        is_gpu: Whether GPU runtime is enabled
        scala_version: Scala version
        spark_version: Apache Spark version
    """
    version: Optional[str] = None
    major_version: Optional[int] = None
    minor_version: Optional[int] = None
    is_ml: bool = False
    is_photon: bool = False
    is_gpu: bool = False
    scala_version: Optional[str] = None
    spark_version: Optional[str] = None


class DatabricksContext:
    """
    Context manager for Databricks environment.

    Provides utilities for detecting the runtime environment,
    accessing cluster information, and managing Databricks-specific
    resources.

    Example:
        >>> ctx = DatabricksContext()
        >>>
        >>> # Check environment
        >>> if ctx.is_databricks:
        ...     print(f"Cluster: {ctx.cluster_info.cluster_name}")
        ...     print(f"Runtime: {ctx.runtime_info.version}")
        >>>
        >>> # Access secrets
        >>> api_key = ctx.get_secret("my-scope", "api-key")
        >>>
        >>> # Get Spark session
        >>> spark = ctx.get_spark_session()
    """

    def __init__(self, spark: Optional["SparkSession"] = None) -> None:
        """
        Initialize Databricks context.

        Args:
            spark: Optional SparkSession (auto-detected if not provided)
        """
        self._spark = spark
        self._dbutils = None
        self._cluster_info: Optional[ClusterInfo] = None
        self._runtime_info: Optional[RuntimeInfo] = None

    @property
    def is_databricks(self) -> bool:
        """Check if running on Databricks."""
        # Multiple detection methods for reliability
        indicators = [
            "DATABRICKS_RUNTIME_VERSION" in os.environ,
            "SPARK_HOME" in os.environ and "databricks" in os.environ.get("SPARK_HOME", "").lower(),
            os.path.exists("/databricks"),
        ]
        return any(indicators)

    @property
    def is_notebook(self) -> bool:
        """Check if running in a Databricks notebook."""
        try:
            # IPython detection
            from IPython import get_ipython
            ipython = get_ipython()
            if ipython is None:
                return False
            return "databricks" in str(type(ipython)).lower()
        except ImportError:
            return False

    @property
    def is_job(self) -> bool:
        """Check if running as a Databricks job."""
        return self.is_databricks and not self.is_notebook

    @property
    def runtime_version(self) -> Optional[str]:
        """Get Databricks Runtime version string."""
        return os.environ.get("DATABRICKS_RUNTIME_VERSION")

    @property
    def cluster_info(self) -> ClusterInfo:
        """Get cluster information."""
        if self._cluster_info is None:
            self._cluster_info = self._detect_cluster_info()
        return self._cluster_info

    @property
    def runtime_info(self) -> RuntimeInfo:
        """Get runtime information."""
        if self._runtime_info is None:
            self._runtime_info = self._detect_runtime_info()
        return self._runtime_info

    def get_spark_session(self) -> "SparkSession":
        """
        Get or create SparkSession.

        Returns:
            Active SparkSession

        Note:
            On Databricks, returns the pre-configured session.
            Locally, creates a new session if needed.
        """
        if self._spark is not None:
            return self._spark

        from pyspark.sql import SparkSession

        if self.is_databricks:
            # Get existing Databricks session
            self._spark = SparkSession.builder.getOrCreate()
        else:
            # Create local session
            self._spark = (
                SparkSession.builder
                .appName("DataForge")
                .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
                .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
                .getOrCreate()
            )

        return self._spark

    def get_dbutils(self) -> Any:
        """
        Get DBUtils instance.

        Returns:
            DBUtils instance for Databricks utilities

        Raises:
            RuntimeError: If not running on Databricks
        """
        if not self.is_databricks:
            raise RuntimeError("DBUtils is only available on Databricks")

        if self._dbutils is None:
            try:
                # Try notebook method first
                from pyspark.dbutils import DBUtils
                self._dbutils = DBUtils(self.get_spark_session())
            except ImportError:
                try:
                    # Fallback for jobs
                    import IPython
                    self._dbutils = IPython.get_ipython().user_ns.get("dbutils")
                except Exception:
                    raise RuntimeError("Could not access DBUtils")

        return self._dbutils

    def get_secret(self, scope: str, key: str) -> str:
        """
        Get secret from Databricks secret scope.

        Args:
            scope: Secret scope name
            key: Secret key name

        Returns:
            Secret value as string

        Example:
            >>> api_key = ctx.get_secret("production", "api-key")
        """
        dbutils = self.get_dbutils()
        return dbutils.secrets.get(scope=scope, key=key)

    def list_secrets(self, scope: str) -> List[Dict[str, str]]:
        """
        List secrets in a scope.

        Args:
            scope: Secret scope name

        Returns:
            List of secret metadata (key names only, not values)
        """
        dbutils = self.get_dbutils()
        secrets = dbutils.secrets.list(scope=scope)
        return [{"key": s.key} for s in secrets]

    def list_secret_scopes(self) -> List[Dict[str, str]]:
        """
        List all available secret scopes.

        Returns:
            List of scope metadata
        """
        dbutils = self.get_dbutils()
        scopes = dbutils.secrets.listScopes()
        return [{"name": s.name} for s in scopes]

    def get_notebook_path(self) -> Optional[str]:
        """
        Get current notebook path.

        Returns:
            Notebook path or None if not in notebook
        """
        if not self.is_notebook:
            return None

        try:
            dbutils = self.get_dbutils()
            return dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
        except Exception:
            return None

    def get_job_id(self) -> Optional[str]:
        """
        Get current job ID if running as a job.

        Returns:
            Job ID or None
        """
        if not self.is_job:
            return None
        return os.environ.get("DATABRICKS_JOB_ID")

    def get_run_id(self) -> Optional[str]:
        """
        Get current run ID if running as a job.

        Returns:
            Run ID or None
        """
        if not self.is_job:
            return None
        return os.environ.get("DATABRICKS_RUN_ID")

    def set_spark_config(self, key: str, value: str) -> None:
        """
        Set Spark configuration.

        Args:
            key: Configuration key
            value: Configuration value
        """
        spark = self.get_spark_session()
        spark.conf.set(key, value)
        logger.info(f"Set Spark config: {key}={value}")

    def get_spark_config(self, key: str) -> Optional[str]:
        """
        Get Spark configuration value.

        Args:
            key: Configuration key

        Returns:
            Configuration value or None
        """
        spark = self.get_spark_session()
        try:
            return spark.conf.get(key)
        except Exception:
            return None

    def _detect_cluster_info(self) -> ClusterInfo:
        """Detect cluster information from environment."""
        info = ClusterInfo()

        if not self.is_databricks:
            return info

        # Get from environment variables
        info.cluster_id = os.environ.get("DATABRICKS_CLUSTER_ID")
        info.spark_version = os.environ.get("DATABRICKS_RUNTIME_VERSION")

        # Try to get more info from Spark conf
        try:
            spark = self.get_spark_session()
            conf = spark.sparkContext.getConf()

            info.cluster_name = conf.get("spark.databricks.clusterUsageTags.clusterName")
            info.node_type = conf.get("spark.databricks.clusterUsageTags.clusterNodeType")
            info.driver_node_type = conf.get("spark.databricks.clusterUsageTags.driverNodeType")

            # Parse worker count
            workers = conf.get("spark.databricks.clusterUsageTags.clusterWorkers")
            if workers:
                info.num_workers = int(workers)

            # Parse autoscale settings
            min_workers = conf.get("spark.databricks.clusterUsageTags.clusterMinWorkers")
            max_workers = conf.get("spark.databricks.clusterUsageTags.clusterMaxWorkers")
            if min_workers:
                info.autoscale_min = int(min_workers)
            if max_workers:
                info.autoscale_max = int(max_workers)

        except Exception as e:
            logger.debug(f"Could not get full cluster info: {e}")

        return info

    def _detect_runtime_info(self) -> RuntimeInfo:
        """Detect runtime information."""
        info = RuntimeInfo()

        version_str = self.runtime_version
        if not version_str:
            return info

        info.version = version_str

        # Parse version string (e.g., "13.3.x-scala2.12" or "13.3.x-photon-scala2.12")
        try:
            parts = version_str.split("-")
            version_parts = parts[0].split(".")

            info.major_version = int(version_parts[0])
            info.minor_version = int(version_parts[1]) if len(version_parts) > 1 else 0

            # Check for special runtimes
            info.is_photon = "photon" in version_str.lower()
            info.is_ml = "ml" in version_str.lower()
            info.is_gpu = "gpu" in version_str.lower()

            # Extract Scala version
            for part in parts:
                if part.startswith("scala"):
                    info.scala_version = part.replace("scala", "")

        except Exception as e:
            logger.debug(f"Could not parse runtime version: {e}")

        # Get Spark version
        try:
            spark = self.get_spark_session()
            info.spark_version = spark.version
        except Exception:
            pass

        return info

    def require_runtime_version(self, min_version: str) -> None:
        """
        Ensure minimum runtime version.

        Args:
            min_version: Minimum required version (e.g., "13.0")

        Raises:
            RuntimeError: If version requirement not met
        """
        if not self.is_databricks:
            logger.warning("Not running on Databricks, version check skipped")
            return

        runtime = self.runtime_info
        if runtime.major_version is None:
            raise RuntimeError("Could not determine runtime version")

        min_parts = min_version.split(".")
        min_major = int(min_parts[0])
        min_minor = int(min_parts[1]) if len(min_parts) > 1 else 0

        current = (runtime.major_version, runtime.minor_version or 0)
        required = (min_major, min_minor)

        if current < required:
            raise RuntimeError(
                f"Databricks Runtime {min_version}+ required. "
                f"Current version: {runtime.version}"
            )

    def display(self, df: Any) -> None:
        """
        Display DataFrame using Databricks display().

        Falls back to print() if not on Databricks.

        Args:
            df: DataFrame to display
        """
        if self.is_databricks and self.is_notebook:
            try:
                dbutils = self.get_dbutils()
                # Use IPython display
                from IPython.display import display as ipy_display
                ipy_display(df)
            except Exception:
                print(df)
        else:
            print(df)


def get_context() -> DatabricksContext:
    """
    Get global Databricks context.

    Returns:
        DatabricksContext singleton
    """
    global _context
    if "_context" not in globals():
        _context = DatabricksContext()
    return _context
