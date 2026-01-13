"""
DataForge Hardware Detection Module

This module provides utilities for detecting available hardware capabilities
including GPU presence, memory, and cluster availability.

Detection Capabilities:
    - GPU availability and memory (NVIDIA CUDA)
    - System RAM
    - CPU core count
    - Spark cluster availability
    - Databricks environment detection

Example:
    >>> from dataforge.advisor import HardwareDetector
    >>>
    >>> detector = HardwareDetector()
    >>> info = detector.detect_all()
    >>>
    >>> print(f"GPU Available: {info.gpu_available}")
    >>> print(f"GPU Memory: {info.gpu_memory_gb} GB")
    >>> print(f"System RAM: {info.system_memory_gb} GB")
    >>> print(f"Spark Available: {info.spark_available}")
"""

import os
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class HardwareInfo:
    """
    Container for detected hardware information.

    Attributes:
        gpu_available: Whether CUDA GPU is available
        gpu_count: Number of available GPUs
        gpu_memory_gb: GPU memory in gigabytes (per GPU)
        gpu_name: GPU model name
        system_memory_gb: System RAM in gigabytes
        cpu_count: Number of CPU cores
        spark_available: Whether Spark is available
        spark_cluster: Whether running on a Spark cluster
        databricks_runtime: Databricks runtime version if applicable
        is_databricks: Whether running in Databricks
    """
    gpu_available: bool = False
    gpu_count: int = 0
    gpu_memory_gb: float = 0.0
    gpu_name: Optional[str] = None
    system_memory_gb: float = 0.0
    cpu_count: int = 1
    spark_available: bool = False
    spark_cluster: bool = False
    databricks_runtime: Optional[str] = None
    is_databricks: bool = False


class HardwareDetector:
    """
    Hardware capability detector.

    Detects available hardware resources to inform engine selection
    and configuration recommendations.

    Example:
        >>> detector = HardwareDetector()
        >>>
        >>> # Get all hardware info
        >>> info = detector.detect_all()
        >>>
        >>> # Check specific capabilities
        >>> if detector.has_gpu():
        ...     print(f"GPU: {detector.get_gpu_info()}")
    """

    _cached_info: Optional[HardwareInfo] = None

    def detect_all(self, use_cache: bool = True) -> HardwareInfo:
        """
        Detect all hardware capabilities.

        Args:
            use_cache: Use cached results if available

        Returns:
            HardwareInfo with detected capabilities
        """
        if use_cache and self._cached_info is not None:
            return self._cached_info

        info = HardwareInfo(
            gpu_available=self._detect_gpu_available(),
            gpu_count=self._detect_gpu_count(),
            gpu_memory_gb=self._detect_gpu_memory(),
            gpu_name=self._detect_gpu_name(),
            system_memory_gb=self._detect_system_memory(),
            cpu_count=self._detect_cpu_count(),
            spark_available=self._detect_spark_available(),
            spark_cluster=self._detect_spark_cluster(),
            databricks_runtime=self._detect_databricks_runtime(),
            is_databricks=self._detect_is_databricks(),
        )

        self._cached_info = info
        return info

    def has_gpu(self) -> bool:
        """Check if GPU is available."""
        return self._detect_gpu_available()

    def has_spark(self) -> bool:
        """Check if Spark is available."""
        return self._detect_spark_available()

    def is_databricks(self) -> bool:
        """Check if running in Databricks environment."""
        return self._detect_is_databricks()

    def get_gpu_memory_gb(self) -> float:
        """Get GPU memory in GB."""
        return self._detect_gpu_memory()

    def get_system_memory_gb(self) -> float:
        """Get system memory in GB."""
        return self._detect_system_memory()

    # ==========================================================================
    # PRIVATE DETECTION METHODS
    # ==========================================================================

    def _detect_gpu_available(self) -> bool:
        """Detect if CUDA GPU is available."""
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi"],
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
            return False

    def _detect_gpu_count(self) -> int:
        """Detect number of available GPUs."""
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "--list-gpus"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                return len(result.stdout.strip().split("\n"))
        except Exception:
            pass

        # Try using pynvml
        try:
            import pynvml
            pynvml.nvmlInit()
            count = pynvml.nvmlDeviceGetCount()
            pynvml.nvmlShutdown()
            return count
        except Exception:
            pass

        return 0

    def _detect_gpu_memory(self) -> float:
        """Detect GPU memory in GB."""
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                # Get first GPU's memory (in MiB)
                memory_mib = float(result.stdout.strip().split("\n")[0])
                return memory_mib / 1024  # Convert to GB
        except Exception:
            pass

        # Try using pynvml
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            pynvml.nvmlShutdown()
            return info.total / (1024 ** 3)  # Bytes to GB
        except Exception:
            pass

        return 0.0

    def _detect_gpu_name(self) -> Optional[str]:
        """Detect GPU model name."""
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                return result.stdout.strip().split("\n")[0]
        except Exception:
            pass

        return None

    def _detect_system_memory(self) -> float:
        """Detect system memory in GB."""
        try:
            import psutil
            return psutil.virtual_memory().total / (1024 ** 3)
        except ImportError:
            pass

        # Fallback for Linux
        try:
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        # Value in kB
                        kb = int(line.split()[1])
                        return kb / (1024 ** 2)  # kB to GB
        except Exception:
            pass

        # Fallback for macOS
        try:
            import subprocess
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                return int(result.stdout.strip()) / (1024 ** 3)
        except Exception:
            pass

        return 8.0  # Default assumption

    def _detect_cpu_count(self) -> int:
        """Detect number of CPU cores."""
        try:
            import psutil
            return psutil.cpu_count(logical=True) or os.cpu_count() or 1
        except ImportError:
            return os.cpu_count() or 1

    def _detect_spark_available(self) -> bool:
        """Detect if PySpark is available."""
        try:
            from pyspark.sql import SparkSession
            return True
        except ImportError:
            return False

    def _detect_spark_cluster(self) -> bool:
        """Detect if running on a Spark cluster."""
        # Check for cluster-related environment variables
        cluster_indicators = [
            "SPARK_HOME",
            "SPARK_MASTER",
            "YARN_CONF_DIR",
            "MESOS_NATIVE_LIBRARY",
            "KUBERNETES_SERVICE_HOST",
        ]

        for var in cluster_indicators:
            if os.environ.get(var):
                return True

        # Check for Databricks cluster
        if self._detect_is_databricks():
            return True

        return False

    def _detect_databricks_runtime(self) -> Optional[str]:
        """Detect Databricks runtime version."""
        # Check for Databricks runtime version
        runtime = os.environ.get("DATABRICKS_RUNTIME_VERSION")
        if runtime:
            return runtime

        # Try to get from spark conf
        try:
            from pyspark.sql import SparkSession
            spark = SparkSession.builder.getOrCreate()
            runtime = spark.conf.get("spark.databricks.clusterUsageTags.clusterName", None)
            if runtime:
                return "databricks"
        except Exception:
            pass

        return None

    def _detect_is_databricks(self) -> bool:
        """Detect if running in Databricks environment."""
        # Check environment variables
        databricks_indicators = [
            "DATABRICKS_RUNTIME_VERSION",
            "DB_HOME",
            "DATABRICKS_GIT_PROVIDER",
        ]

        for var in databricks_indicators:
            if os.environ.get(var):
                return True

        # Check for /databricks directory (common in Databricks)
        if os.path.exists("/databricks"):
            return True

        return False

    def clear_cache(self) -> None:
        """Clear cached hardware information."""
        self._cached_info = None
