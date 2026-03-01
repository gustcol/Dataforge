"""
DataForge Engine Recommender Module

This module provides intelligent engine selection based on data characteristics,
available hardware, and performance requirements.

Decision Matrix:
    The recommender uses a comprehensive decision matrix considering:
    - Dataset size (MB/GB)
    - GPU availability and memory
    - Spark cluster availability
    - Operation complexity
    - Memory constraints

Recommendation Logic:
    1. < 100 MB: Always Pandas (minimal overhead)
    2. 100 MB - 1 GB:
       - With GPU: RAPIDS (5-20x faster)
       - Without: Pandas (simpler)
    3. 1 GB - 10 GB:
       - With GPU (16GB+): RAPIDS (best performance)
       - With Cluster: Spark (distributed)
       - Without: Spark local mode
    4. > 10 GB:
       - With Cluster: Spark (required for scale)
       - With GPU (32GB+): RAPIDS chunked (if fits in chunks)
       - Otherwise: Spark local (may be slow)

Example:
    >>> from dataforge.advisor import EngineRecommender
    >>>
    >>> recommender = EngineRecommender()
    >>>
    >>> # Get recommendation for file
    >>> rec = recommender.recommend_for_path("large_data.parquet")
    >>> print(f"Recommended: {rec.engine}")
    >>> print(f"Reason: {rec.reason}")
    >>> print(f"Expected performance: {rec.performance_notes}")
    >>>
    >>> # Get recommendation by size
    >>> rec = recommender.recommend(data_size_mb=5000, has_gpu=True)
    >>> print(rec)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union, List

from dataforge.core.base import EngineType
from dataforge.core.config import EngineConfig
from dataforge.advisor.size_analyzer import SizeAnalyzer
from dataforge.advisor.hardware_detector import HardwareDetector, HardwareInfo


@dataclass
class EngineRecommendation:
    """
    Engine recommendation result.

    Attributes:
        engine: Recommended engine type
        reason: Human-readable explanation
        confidence: Confidence score (0.0 - 1.0)
        performance_notes: Expected performance characteristics
        alternatives: Alternative engines to consider
        warnings: Any warnings or caveats
    """
    engine: EngineType
    reason: str
    confidence: float = 1.0
    performance_notes: Optional[str] = None
    alternatives: Optional[List[EngineType]] = None
    warnings: Optional[List[str]] = None

    def __str__(self) -> str:
        return f"EngineRecommendation({self.engine.value}): {self.reason}"


class EngineRecommender:
    """
    Intelligent engine selection advisor.

    Analyzes data characteristics and hardware capabilities to recommend
    the optimal processing engine.

    Configuration:
        The recommender can be customized via EngineConfig to adjust:
        - Size thresholds for each engine
        - GPU preference settings
        - Cluster detection behavior

    Example:
        >>> from dataforge.advisor import EngineRecommender
        >>> from dataforge.core import EngineConfig
        >>>
        >>> # Default configuration
        >>> recommender = EngineRecommender()
        >>>
        >>> # Custom thresholds
        >>> config = EngineConfig(
        ...     pandas_max_size_mb=2048,  # Allow larger pandas operations
        ...     prefer_gpu=True
        ... )
        >>> recommender = EngineRecommender(config)
    """

    def __init__(
        self,
        config: Optional[EngineConfig] = None,
        hardware_info: Optional[HardwareInfo] = None
    ) -> None:
        """
        Initialize engine recommender.

        Args:
            config: Engine configuration with thresholds
            hardware_info: Pre-detected hardware info (auto-detects if None)
        """
        self.config = config or EngineConfig()
        self._hardware_detector = HardwareDetector()
        self._size_analyzer = SizeAnalyzer()

        # Cache hardware info
        if hardware_info:
            self._hardware_info = hardware_info
        else:
            self._hardware_info = None

    @property
    def hardware_info(self) -> HardwareInfo:
        """Get hardware information (cached)."""
        if self._hardware_info is None:
            self._hardware_info = self._hardware_detector.detect_all()
        return self._hardware_info

    def recommend(
        self,
        data_size_mb: Optional[float] = None,
        data_size_bytes: Optional[int] = None,
        has_gpu: Optional[bool] = None,
        has_cluster: Optional[bool] = None,
        operation_type: Optional[str] = None
    ) -> EngineRecommendation:
        """
        Get engine recommendation based on parameters.

        Args:
            data_size_mb: Data size in megabytes
            data_size_bytes: Data size in bytes (alternative to MB)
            has_gpu: Override GPU detection
            has_cluster: Override cluster detection
            operation_type: Type of operation ('aggregation', 'join', 'transform')

        Returns:
            EngineRecommendation with suggested engine

        Example:
            >>> rec = recommender.recommend(data_size_mb=5000)
            >>> rec = recommender.recommend(data_size_bytes=5_000_000_000)
        """
        # Convert bytes to MB if needed
        if data_size_mb is None and data_size_bytes is not None:
            data_size_mb = data_size_bytes / (1024 * 1024)
        elif data_size_mb is None:
            data_size_mb = 100  # Default assumption

        # Get hardware capabilities
        gpu_available = has_gpu if has_gpu is not None else self.hardware_info.gpu_available
        gpu_memory_gb = self.hardware_info.gpu_memory_gb if gpu_available else 0
        cluster_available = has_cluster if has_cluster is not None else (
            self.config.cluster_available or self.hardware_info.spark_cluster
        )

        # Decision logic
        return self._make_recommendation(
            data_size_mb=data_size_mb,
            gpu_available=gpu_available,
            gpu_memory_gb=gpu_memory_gb,
            cluster_available=cluster_available,
            operation_type=operation_type
        )

    def recommend_for_path(
        self,
        path: Union[str, Path],
        operation_type: Optional[str] = None
    ) -> EngineRecommendation:
        """
        Get engine recommendation for a file or directory.

        Analyzes the file/directory to determine size and characteristics.

        Args:
            path: Path to data file or directory
            operation_type: Type of operation to perform

        Returns:
            EngineRecommendation with suggested engine

        Example:
            >>> rec = recommender.recommend_for_path("data.parquet")
            >>> rec = recommender.recommend_for_path("/data/partitioned/")
        """
        path = Path(path)

        if not path.exists():
            # Can't analyze, return default
            return EngineRecommendation(
                engine=EngineType.PANDAS,
                reason="Path does not exist, using default engine",
                confidence=0.5,
                warnings=["Could not analyze path - file does not exist"]
            )

        # Analyze data size
        size_info = self._size_analyzer.analyze_path(path)

        # Get recommendation
        rec = self.recommend(
            data_size_mb=size_info.file_size_mb,
            operation_type=operation_type
        )

        # Add file-specific notes
        if size_info.format:
            rec.performance_notes = (
                f"{rec.performance_notes or ''} "
                f"Format: {size_info.format}, Files: {size_info.file_count}"
            ).strip()

        return rec

    def _make_recommendation(
        self,
        data_size_mb: float,
        gpu_available: bool,
        gpu_memory_gb: float,
        cluster_available: bool,
        operation_type: Optional[str]
    ) -> EngineRecommendation:
        """
        Core recommendation logic.

        Decision Tree:
        1. Very small data (< 100MB) -> Pandas
        2. Small data (100MB - 1GB):
           - GPU available -> RAPIDS
           - Otherwise -> Pandas
        3. Medium data (1GB - 10GB):
           - GPU with sufficient memory -> RAPIDS
           - Cluster available -> Spark
           - Otherwise -> Spark local
        4. Large data (> 10GB):
           - Cluster available -> Spark (required)
           - Large GPU (32GB+) -> RAPIDS chunked
           - Otherwise -> Spark local (with warnings)
        """
        warnings = []
        alternatives = []

        # Size thresholds
        pandas_max = self.config.pandas_max_size_mb
        rapids_min = self.config.rapids_min_size_mb
        rapids_max = self.config.rapids_max_size_mb
        spark_min = self.config.spark_min_size_mb

        # Very small data: Always Pandas
        if data_size_mb < rapids_min:
            return EngineRecommendation(
                engine=EngineType.PANDAS,
                reason=f"Data size ({data_size_mb:.1f}MB) is small - Pandas is most efficient",
                confidence=1.0,
                performance_notes="Pandas has minimal overhead for small datasets",
                alternatives=[EngineType.POLARS] if data_size_mb > 50 else None
            )

        # Small data (100MB - 1GB)
        if data_size_mb < pandas_max:
            if gpu_available and self.config.prefer_gpu:
                return EngineRecommendation(
                    engine=EngineType.RAPIDS,
                    reason=f"GPU available - RAPIDS provides 5-20x speedup for {data_size_mb:.1f}MB data",
                    confidence=0.9,
                    performance_notes="Expected 5-20x performance improvement over Pandas",
                    alternatives=[EngineType.POLARS]
                )
            else:
                return EngineRecommendation(
                    engine=EngineType.POLARS,
                    reason=f"Data size ({data_size_mb:.1f}MB) benefits from Polars' Rust-backed performance",
                    confidence=0.85,
                    performance_notes="Polars provides 5-20x speedup over Pandas with lazy evaluation",
                    alternatives=[EngineType.PANDAS, EngineType.SPARK] if cluster_available else [EngineType.PANDAS]
                )

        # Medium data (1GB - 10GB)
        if data_size_mb < 10 * 1024:
            # Check if RAPIDS can handle it
            estimated_memory_gb = data_size_mb * 5 / 1024  # Rough estimate

            if gpu_available and self.config.prefer_gpu:
                if gpu_memory_gb >= estimated_memory_gb:
                    return EngineRecommendation(
                        engine=EngineType.RAPIDS,
                        reason=f"GPU ({gpu_memory_gb:.1f}GB) can handle {data_size_mb:.1f}MB data efficiently",
                        confidence=0.9,
                        performance_notes="Expected 10-50x speedup for aggregations and joins",
                        alternatives=[EngineType.POLARS, EngineType.SPARK]
                    )
                else:
                    warnings.append(
                        f"GPU memory ({gpu_memory_gb:.1f}GB) may be insufficient. "
                        "Spilling may occur."
                    )
                    alternatives.append(EngineType.RAPIDS)

            if cluster_available:
                return EngineRecommendation(
                    engine=EngineType.SPARK,
                    reason=f"Data size ({data_size_mb:.1f}MB) benefits from distributed processing",
                    confidence=0.85,
                    performance_notes="Spark provides fault tolerance and scalability",
                    alternatives=[EngineType.POLARS] + (alternatives or []),
                    warnings=warnings or None
                )
            else:
                return EngineRecommendation(
                    engine=EngineType.POLARS,
                    reason=f"Data size ({data_size_mb:.1f}MB) - Polars handles efficiently on single node",
                    confidence=0.80,
                    performance_notes="Polars streaming mode supports out-of-core processing",
                    alternatives=[EngineType.SPARK] + (alternatives or []),
                    warnings=warnings or None
                )

        # Large data (> 10GB)
        data_size_gb = data_size_mb / 1024

        if cluster_available:
            return EngineRecommendation(
                engine=EngineType.SPARK,
                reason=f"Large dataset ({data_size_gb:.1f}GB) requires distributed processing",
                confidence=0.95,
                performance_notes="Spark cluster provides required scalability",
                alternatives=None
            )

        if gpu_available and gpu_memory_gb >= 32:
            return EngineRecommendation(
                engine=EngineType.RAPIDS,
                reason=f"Large GPU ({gpu_memory_gb:.1f}GB) can process {data_size_gb:.1f}GB with chunking",
                confidence=0.7,
                performance_notes="Will use chunked processing - consider Spark cluster for better reliability",
                alternatives=[EngineType.SPARK],
                warnings=["Large dataset may require chunked processing"]
            )

        # Fallback: Spark local with warnings
        return EngineRecommendation(
            engine=EngineType.SPARK,
            reason=f"Large dataset ({data_size_gb:.1f}GB) - using Spark local mode",
            confidence=0.6,
            performance_notes="Performance may be limited without cluster",
            alternatives=None,
            warnings=[
                f"Dataset ({data_size_gb:.1f}GB) is large for local processing",
                "Consider setting up a Spark cluster for production use",
                "Monitor memory usage carefully"
            ]
        )

    def get_performance_estimate(
        self,
        data_size_mb: float,
        engine: EngineType,
        operation: str = "general"
    ) -> dict:
        """
        Estimate performance characteristics for an engine.

        Args:
            data_size_mb: Data size in MB
            engine: Engine to estimate for
            operation: Type of operation

        Returns:
            Dictionary with performance estimates
        """
        estimates = {
            "engine": engine.value,
            "data_size_mb": data_size_mb,
            "operation": operation,
        }

        # Base estimates (relative to Pandas baseline = 1.0)
        if engine == EngineType.PANDAS:
            estimates["relative_speed"] = 1.0
            estimates["memory_overhead"] = "3-5x data size"
            estimates["startup_time"] = "minimal"

        elif engine == EngineType.POLARS:
            if operation in ["aggregation", "groupby"]:
                estimates["relative_speed"] = 15.0
            elif operation in ["join"]:
                estimates["relative_speed"] = 10.0
            else:
                estimates["relative_speed"] = 5.0
            estimates["memory_overhead"] = "2-3x data size (more efficient than Pandas)"
            estimates["startup_time"] = "minimal"

        elif engine == EngineType.SPARK:
            if data_size_mb < 1000:
                estimates["relative_speed"] = 0.5  # Overhead dominates
            else:
                estimates["relative_speed"] = 2.0  # Parallelism helps
            estimates["memory_overhead"] = "distributed across cluster"
            estimates["startup_time"] = "5-30 seconds"

        elif engine == EngineType.RAPIDS:
            if operation in ["aggregation", "groupby"]:
                estimates["relative_speed"] = 50.0
            elif operation in ["join"]:
                estimates["relative_speed"] = 20.0
            else:
                estimates["relative_speed"] = 10.0
            estimates["memory_overhead"] = "GPU memory required"
            estimates["startup_time"] = "1-5 seconds"

        return estimates
