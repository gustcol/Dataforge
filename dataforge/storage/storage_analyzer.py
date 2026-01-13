"""
DataForge Storage Analyzer

Analyze cloud storage for performance optimization opportunities.

Features:
    - File distribution analysis
    - Partition analysis
    - Format detection
    - Size distribution metrics

Author: DataForge Team
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING
import logging

if TYPE_CHECKING:
    from pyspark.sql import SparkSession, DataFrame

logger = logging.getLogger(__name__)


@dataclass
class FileStats:
    """Statistics for a single file."""
    path: str
    size_bytes: int
    format: str
    compression: Optional[str]
    partition_values: Dict[str, str] = field(default_factory=dict)


@dataclass
class StorageReport:
    """Comprehensive storage analysis report."""
    path: str
    total_size_bytes: int
    total_files: int
    file_size_distribution: Dict[str, int]
    format_distribution: Dict[str, int]
    partition_analysis: Dict[str, Any]
    recommendations: List[str]


class StorageAnalyzer:
    """
    Analyze cloud storage paths for optimization.

    Example:
        >>> analyzer = StorageAnalyzer(spark)
        >>> report = analyzer.analyze("s3://bucket/data/")
        >>> print(report.recommendations)
    """

    def __init__(self, spark: Optional["SparkSession"] = None):
        """Initialize analyzer with SparkSession."""
        self.spark = spark

    def _get_spark(self) -> "SparkSession":
        """Get or create SparkSession."""
        if self.spark is None:
            from pyspark.sql import SparkSession
            self.spark = SparkSession.active
        return self.spark

    def analyze(self, path: str) -> StorageReport:
        """
        Analyze storage path.

        Args:
            path: S3/ADLS/GCS path to analyze

        Returns:
            StorageReport with analysis results
        """
        spark = self._get_spark()

        try:
            # Get file listing using binaryFile format
            files_df = spark.read.format("binaryFile").load(path)
            files = files_df.select("path", "length").collect()
        except Exception as e:
            logger.error(f"Failed to analyze path {path}: {e}")
            return StorageReport(
                path=path,
                total_size_bytes=0,
                total_files=0,
                file_size_distribution={},
                format_distribution={},
                partition_analysis={},
                recommendations=[f"Error: {e}"]
            )

        # Calculate statistics
        total_files = len(files)
        total_size = sum(f.length for f in files)

        # Size distribution buckets
        size_dist = {
            "< 1MB": 0,
            "1-10MB": 0,
            "10-100MB": 0,
            "100MB-1GB": 0,
            "> 1GB": 0,
        }

        format_dist: Dict[str, int] = {}

        for f in files:
            size_mb = f.length / (1024 * 1024)

            # Size bucket
            if size_mb < 1:
                size_dist["< 1MB"] += 1
            elif size_mb < 10:
                size_dist["1-10MB"] += 1
            elif size_mb < 100:
                size_dist["10-100MB"] += 1
            elif size_mb < 1024:
                size_dist["100MB-1GB"] += 1
            else:
                size_dist["> 1GB"] += 1

            # Format detection
            fmt = self._detect_format(f.path)
            format_dist[fmt] = format_dist.get(fmt, 0) + 1

        # Generate recommendations
        recommendations = self._generate_recommendations(
            total_files, total_size, size_dist, format_dist
        )

        return StorageReport(
            path=path,
            total_size_bytes=total_size,
            total_files=total_files,
            file_size_distribution=size_dist,
            format_distribution=format_dist,
            partition_analysis={},
            recommendations=recommendations
        )

    def _detect_format(self, path: str) -> str:
        """Detect file format from path."""
        path_lower = path.lower()
        if path_lower.endswith(".parquet"):
            return "parquet"
        elif path_lower.endswith(".orc"):
            return "orc"
        elif ".json" in path_lower:
            return "json"
        elif ".csv" in path_lower:
            return "csv"
        elif path_lower.endswith(".avro"):
            return "avro"
        else:
            return "other"

    def _generate_recommendations(
        self,
        total_files: int,
        total_size: int,
        size_dist: Dict[str, int],
        format_dist: Dict[str, int]
    ) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []

        # Small files issue
        small_files = size_dist.get("< 1MB", 0) + size_dist.get("1-10MB", 0)
        if total_files > 0 and small_files / total_files > 0.3:
            recommendations.append(
                f"SMALL FILES: {small_files} files ({small_files/total_files*100:.1f}%) are under 10MB. "
                "Consider compacting to 128MB-1GB files."
            )

        # Format recommendation
        non_columnar = format_dist.get("csv", 0) + format_dist.get("json", 0)
        if total_files > 0 and non_columnar / total_files > 0.2:
            recommendations.append(
                "FORMAT: Convert CSV/JSON to Parquet or Delta Lake for better performance."
            )

        # Many files
        if total_files > 10000:
            recommendations.append(
                f"FILE COUNT: {total_files:,} files detected. Consider using OPTIMIZE or reducing partitions."
            )

        return recommendations
