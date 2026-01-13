"""
DataForge S3 Optimizer

Comprehensive S3 performance analysis and optimization utilities.

Features:
    - S3 configuration optimization
    - File size analysis and recommendations
    - Partitioning strategy optimization
    - Read/write performance tuning
    - Cost optimization suggestions

Best Practices Implemented:
    1. Optimal file sizes (128MB - 1GB for analytics)
    2. Appropriate partitioning strategies
    3. Efficient S3 request patterns
    4. Compression optimization
    5. Storage class recommendations

Example:
    >>> from dataforge.storage import S3Optimizer
    >>>
    >>> optimizer = S3Optimizer(spark)
    >>>
    >>> # Analyze and get recommendations
    >>> report = optimizer.analyze_path("s3://bucket/data/")
    >>> print(report.recommendations)
    >>>
    >>> # Apply optimal configurations
    >>> optimizer.apply_optimal_config()

Author: DataForge Team
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from enum import Enum
import logging

if TYPE_CHECKING:
    from pyspark.sql import SparkSession, DataFrame

logger = logging.getLogger(__name__)


class S3StorageClass(Enum):
    """S3 storage class options."""
    STANDARD = "STANDARD"
    STANDARD_IA = "STANDARD_IA"
    ONEZONE_IA = "ONEZONE_IA"
    INTELLIGENT_TIERING = "INTELLIGENT_TIERING"
    GLACIER = "GLACIER"
    GLACIER_INSTANT = "GLACIER_INSTANT_RETRIEVAL"
    DEEP_ARCHIVE = "DEEP_ARCHIVE"


class CompressionType(Enum):
    """Supported compression types."""
    NONE = "none"
    SNAPPY = "snappy"
    GZIP = "gzip"
    LZ4 = "lz4"
    ZSTD = "zstd"
    BROTLI = "brotli"


@dataclass
class S3Config:
    """
    S3 configuration settings for optimal performance.

    Attributes:
        max_connections: Maximum concurrent connections to S3
        multipart_threshold: Size threshold for multipart uploads (bytes)
        multipart_chunksize: Chunk size for multipart uploads (bytes)
        max_bandwidth: Maximum bandwidth limit (MB/s, None for unlimited)
        use_accelerate: Use S3 Transfer Acceleration
        use_path_style: Use path-style addressing
        ssl_enabled: Enable SSL for transfers
        retry_max_attempts: Maximum retry attempts
        connect_timeout: Connection timeout in seconds
        read_timeout: Read timeout in seconds

    Example:
        >>> config = S3Config(
        ...     max_connections=200,
        ...     multipart_threshold=100 * 1024 * 1024,  # 100MB
        ...     use_accelerate=True
        ... )
    """
    max_connections: int = 100
    multipart_threshold: int = 64 * 1024 * 1024  # 64MB
    multipart_chunksize: int = 64 * 1024 * 1024  # 64MB
    max_bandwidth: Optional[int] = None
    use_accelerate: bool = False
    use_path_style: bool = False
    ssl_enabled: bool = True
    retry_max_attempts: int = 10
    connect_timeout: int = 60
    read_timeout: int = 60


@dataclass
class S3PerformanceIssue:
    """
    Represents a performance issue found in S3 storage.

    Attributes:
        severity: Issue severity (high, medium, low)
        category: Issue category (file_size, partitioning, format, etc.)
        description: Detailed description of the issue
        recommendation: Suggested fix
        estimated_improvement: Estimated performance improvement
        affected_paths: List of affected S3 paths
    """
    severity: str
    category: str
    description: str
    recommendation: str
    estimated_improvement: Optional[str] = None
    affected_paths: List[str] = field(default_factory=list)


@dataclass
class S3PerformanceReport:
    """
    Comprehensive S3 performance analysis report.

    Attributes:
        path: Analyzed S3 path
        total_size_gb: Total size in GB
        total_files: Total number of files
        avg_file_size_mb: Average file size in MB
        min_file_size_mb: Minimum file size in MB
        max_file_size_mb: Maximum file size in MB
        file_formats: Dictionary of file formats and counts
        compression_types: Dictionary of compression types and counts
        partition_columns: Detected partition columns
        issues: List of performance issues found
        recommendations: List of optimization recommendations
        estimated_cost_savings: Estimated monthly cost savings
        overall_score: Performance score (0-100)
    """
    path: str
    total_size_gb: float
    total_files: int
    avg_file_size_mb: float
    min_file_size_mb: float
    max_file_size_mb: float
    file_formats: Dict[str, int]
    compression_types: Dict[str, int]
    partition_columns: List[str]
    issues: List[S3PerformanceIssue]
    recommendations: List[str]
    estimated_cost_savings: Optional[float] = None
    overall_score: int = 0


class S3Optimizer:
    """
    S3 performance optimizer for Spark workloads.

    Provides analysis, recommendations, and configuration optimization
    for S3-based data lakes.

    Example:
        >>> optimizer = S3Optimizer(spark)
        >>>
        >>> # Analyze path
        >>> report = optimizer.analyze_path("s3://bucket/data/")
        >>> print(f"Performance Score: {report.overall_score}/100")
        >>> for issue in report.issues:
        ...     print(f"[{issue.severity}] {issue.description}")
        >>>
        >>> # Apply optimizations
        >>> optimizer.apply_optimal_config()
    """

    # Optimal file size range for analytics (128MB - 1GB)
    OPTIMAL_MIN_FILE_SIZE_MB = 128
    OPTIMAL_MAX_FILE_SIZE_MB = 1024

    # Small file threshold (files below this are problematic)
    SMALL_FILE_THRESHOLD_MB = 32

    # Large file threshold (files above this may cause issues)
    LARGE_FILE_THRESHOLD_MB = 2048

    def __init__(
        self,
        spark: Optional["SparkSession"] = None,
        config: Optional[S3Config] = None
    ):
        """
        Initialize S3 Optimizer.

        Args:
            spark: SparkSession (optional, will get active session if not provided)
            config: S3 configuration settings
        """
        self.spark = spark
        self.config = config or S3Config()
        self._hadoop_conf = None

    def _get_spark(self) -> "SparkSession":
        """Get or create SparkSession."""
        if self.spark is None:
            from pyspark.sql import SparkSession
            self.spark = SparkSession.active
        return self.spark

    def _get_hadoop_conf(self):
        """Get Hadoop configuration."""
        if self._hadoop_conf is None:
            self._hadoop_conf = self._get_spark()._jsc.hadoopConfiguration()
        return self._hadoop_conf

    # =========================================================================
    # CONFIGURATION OPTIMIZATION
    # =========================================================================

    def apply_optimal_config(self, workload_type: str = "analytics") -> Dict[str, str]:
        """
        Apply optimal S3 configuration for Spark.

        Args:
            workload_type: Type of workload ("analytics", "etl", "streaming")

        Returns:
            Dictionary of applied configurations

        Best Practices Applied:
            - Optimized connection pooling
            - Efficient multipart upload settings
            - Appropriate timeouts and retries
            - Fast upload/download algorithms
        """
        spark = self._get_spark()
        configs = {}

        # Base S3A configurations
        base_configs = {
            # Connection settings
            "spark.hadoop.fs.s3a.connection.maximum": str(self.config.max_connections),
            "spark.hadoop.fs.s3a.connection.timeout": f"{self.config.connect_timeout}000",
            "spark.hadoop.fs.s3a.socket.timeout": f"{self.config.read_timeout}000",

            # Retry settings
            "spark.hadoop.fs.s3a.attempts.maximum": str(self.config.retry_max_attempts),
            "spark.hadoop.fs.s3a.retry.limit": str(self.config.retry_max_attempts),

            # Multipart upload settings
            "spark.hadoop.fs.s3a.multipart.threshold": str(self.config.multipart_threshold),
            "spark.hadoop.fs.s3a.multipart.size": str(self.config.multipart_chunksize),

            # Fast upload for better write performance
            "spark.hadoop.fs.s3a.fast.upload": "true",
            "spark.hadoop.fs.s3a.fast.upload.buffer": "disk",
            "spark.hadoop.fs.s3a.fast.upload.active.blocks": "8",

            # Path style access (for compatibility)
            "spark.hadoop.fs.s3a.path.style.access": str(self.config.use_path_style).lower(),

            # SSL
            "spark.hadoop.fs.s3a.connection.ssl.enabled": str(self.config.ssl_enabled).lower(),

            # Committer for better write consistency
            "spark.hadoop.fs.s3a.committer.name": "magic",
            "spark.hadoop.fs.s3a.committer.magic.enabled": "true",

            # Block size optimization
            "spark.hadoop.fs.s3a.block.size": "128M",

            # Read ahead for better sequential read performance
            "spark.hadoop.fs.s3a.readahead.range": "64K",
        }

        # Workload-specific optimizations
        if workload_type == "analytics":
            base_configs.update({
                "spark.hadoop.fs.s3a.experimental.input.fadvise": "random",
                "spark.sql.files.maxPartitionBytes": "128m",
                "spark.sql.files.openCostInBytes": "4m",
            })
        elif workload_type == "etl":
            base_configs.update({
                "spark.hadoop.fs.s3a.experimental.input.fadvise": "sequential",
                "spark.sql.files.maxPartitionBytes": "256m",
                "spark.hadoop.fs.s3a.threads.max": "64",
            })
        elif workload_type == "streaming":
            base_configs.update({
                "spark.hadoop.fs.s3a.connection.maximum": "200",
                "spark.hadoop.fs.s3a.threads.max": "32",
                "spark.sql.streaming.stateStore.providerClass":
                    "org.apache.spark.sql.execution.streaming.state.RocksDBStateStoreProvider",
            })

        # Apply configurations
        for key, value in base_configs.items():
            spark.conf.set(key, value)
            configs[key] = value

        logger.info(f"Applied {len(configs)} S3 optimizations for {workload_type} workload")
        return configs

    def get_current_config(self) -> Dict[str, str]:
        """
        Get current S3-related Spark configurations.

        Returns:
            Dictionary of current S3 configurations
        """
        spark = self._get_spark()
        s3_configs = {}

        all_configs = spark.sparkContext.getConf().getAll()
        for key, value in all_configs:
            if "s3" in key.lower() or "fs.s3a" in key:
                s3_configs[key] = value

        return s3_configs

    # =========================================================================
    # PATH ANALYSIS
    # =========================================================================

    def analyze_path(self, path: str) -> S3PerformanceReport:
        """
        Analyze S3 path for performance issues.

        Args:
            path: S3 path to analyze (s3://bucket/prefix/)

        Returns:
            S3PerformanceReport with analysis and recommendations

        Example:
            >>> report = optimizer.analyze_path("s3://my-bucket/data/")
            >>> print(f"Score: {report.overall_score}/100")
            >>> print(f"Issues: {len(report.issues)}")
            >>> for rec in report.recommendations:
            ...     print(f"  - {rec}")
        """
        spark = self._get_spark()

        # Get file listing
        try:
            files_df = spark.read.format("binaryFile").load(path)
            file_stats = files_df.selectExpr(
                "path",
                "length as size",
                "modificationTime"
            ).collect()
        except Exception as e:
            logger.warning(f"Could not read path {path}: {e}")
            # Return empty report
            return S3PerformanceReport(
                path=path,
                total_size_gb=0,
                total_files=0,
                avg_file_size_mb=0,
                min_file_size_mb=0,
                max_file_size_mb=0,
                file_formats={},
                compression_types={},
                partition_columns=[],
                issues=[S3PerformanceIssue(
                    severity="high",
                    category="access",
                    description=f"Could not access path: {e}",
                    recommendation="Check S3 permissions and path validity"
                )],
                recommendations=["Verify S3 path and permissions"],
                overall_score=0
            )

        # Calculate statistics
        total_files = len(file_stats)
        if total_files == 0:
            return S3PerformanceReport(
                path=path,
                total_size_gb=0,
                total_files=0,
                avg_file_size_mb=0,
                min_file_size_mb=0,
                max_file_size_mb=0,
                file_formats={},
                compression_types={},
                partition_columns=[],
                issues=[],
                recommendations=["No files found in path"],
                overall_score=100
            )

        sizes = [f.size for f in file_stats]
        total_size_bytes = sum(sizes)
        total_size_gb = total_size_bytes / (1024 ** 3)
        avg_size_mb = (total_size_bytes / total_files) / (1024 ** 2)
        min_size_mb = min(sizes) / (1024 ** 2)
        max_size_mb = max(sizes) / (1024 ** 2)

        # Analyze file formats
        file_formats = {}
        compression_types = {}
        for f in file_stats:
            path_lower = f.path.lower()
            # Detect format
            if path_lower.endswith(".parquet"):
                fmt = "parquet"
            elif path_lower.endswith(".orc"):
                fmt = "orc"
            elif path_lower.endswith(".json") or path_lower.endswith(".json.gz"):
                fmt = "json"
            elif path_lower.endswith(".csv") or path_lower.endswith(".csv.gz"):
                fmt = "csv"
            elif path_lower.endswith(".avro"):
                fmt = "avro"
            else:
                fmt = "other"
            file_formats[fmt] = file_formats.get(fmt, 0) + 1

            # Detect compression
            if ".snappy" in path_lower:
                comp = "snappy"
            elif ".gz" in path_lower or ".gzip" in path_lower:
                comp = "gzip"
            elif ".lz4" in path_lower:
                comp = "lz4"
            elif ".zstd" in path_lower:
                comp = "zstd"
            elif ".bz2" in path_lower:
                comp = "bzip2"
            else:
                comp = "none/internal"
            compression_types[comp] = compression_types.get(comp, 0) + 1

        # Detect partitions
        partition_columns = self._detect_partitions(path)

        # Find issues
        issues = self._analyze_issues(
            total_files=total_files,
            avg_size_mb=avg_size_mb,
            min_size_mb=min_size_mb,
            max_size_mb=max_size_mb,
            file_formats=file_formats,
            compression_types=compression_types,
            partition_columns=partition_columns,
            sizes=sizes
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(issues, file_formats, avg_size_mb)

        # Calculate score
        score = self._calculate_score(issues, avg_size_mb, file_formats)

        return S3PerformanceReport(
            path=path,
            total_size_gb=total_size_gb,
            total_files=total_files,
            avg_file_size_mb=avg_size_mb,
            min_file_size_mb=min_size_mb,
            max_file_size_mb=max_size_mb,
            file_formats=file_formats,
            compression_types=compression_types,
            partition_columns=partition_columns,
            issues=issues,
            recommendations=recommendations,
            overall_score=score
        )

    def _detect_partitions(self, path: str) -> List[str]:
        """Detect partition columns from path structure."""
        partitions = []
        # Look for patterns like column=value
        import re
        partition_pattern = r'/([a-zA-Z_][a-zA-Z0-9_]*)='
        matches = re.findall(partition_pattern, path)
        return list(set(matches))

    def _analyze_issues(
        self,
        total_files: int,
        avg_size_mb: float,
        min_size_mb: float,
        max_size_mb: float,
        file_formats: Dict[str, int],
        compression_types: Dict[str, int],
        partition_columns: List[str],
        sizes: List[int]
    ) -> List[S3PerformanceIssue]:
        """Analyze and identify performance issues."""
        issues = []

        # Issue: Too many small files
        small_files = sum(1 for s in sizes if s < self.SMALL_FILE_THRESHOLD_MB * 1024 * 1024)
        if small_files > total_files * 0.1:  # More than 10% small files
            issues.append(S3PerformanceIssue(
                severity="high",
                category="file_size",
                description=f"{small_files} files ({small_files/total_files*100:.1f}%) are smaller than {self.SMALL_FILE_THRESHOLD_MB}MB",
                recommendation="Compact small files using OPTIMIZE or repartition. Target file size: 128MB-1GB",
                estimated_improvement="2-10x read performance improvement"
            ))

        # Issue: Average file size too small
        if avg_size_mb < self.OPTIMAL_MIN_FILE_SIZE_MB:
            issues.append(S3PerformanceIssue(
                severity="high",
                category="file_size",
                description=f"Average file size ({avg_size_mb:.1f}MB) is below optimal ({self.OPTIMAL_MIN_FILE_SIZE_MB}MB)",
                recommendation="Increase file size by reducing partition count or using coalesce()",
                estimated_improvement="Up to 5x improvement in read throughput"
            ))

        # Issue: Files too large
        large_files = sum(1 for s in sizes if s > self.LARGE_FILE_THRESHOLD_MB * 1024 * 1024)
        if large_files > 0:
            issues.append(S3PerformanceIssue(
                severity="medium",
                category="file_size",
                description=f"{large_files} files are larger than {self.LARGE_FILE_THRESHOLD_MB}MB",
                recommendation="Split large files for better parallelism. Optimal: 128MB-1GB per file",
                estimated_improvement="Better memory usage and parallelism"
            ))

        # Issue: Using non-columnar format
        row_formats = file_formats.get("csv", 0) + file_formats.get("json", 0)
        if row_formats > total_files * 0.5:
            issues.append(S3PerformanceIssue(
                severity="high",
                category="format",
                description=f"{row_formats/total_files*100:.1f}% of files use row-based formats (CSV/JSON)",
                recommendation="Convert to Parquet or Delta Lake for 2-10x better performance",
                estimated_improvement="2-10x query performance, 3-5x storage reduction"
            ))

        # Issue: No compression or suboptimal compression
        if compression_types.get("none/internal", 0) > total_files * 0.3:
            issues.append(S3PerformanceIssue(
                severity="medium",
                category="compression",
                description="Many files may lack optimal compression",
                recommendation="Use Snappy for balanced speed/compression or ZSTD for better compression",
                estimated_improvement="20-80% storage reduction"
            ))

        # Issue: Using GZIP compression
        if compression_types.get("gzip", 0) > total_files * 0.3:
            issues.append(S3PerformanceIssue(
                severity="low",
                category="compression",
                description="Using GZIP compression which is not splittable",
                recommendation="Use Snappy or LZ4 for Parquet files (splittable, faster)",
                estimated_improvement="Faster decompression, better parallelism"
            ))

        # Issue: Too many partitions (over-partitioning)
        if total_files > 10000:
            issues.append(S3PerformanceIssue(
                severity="medium",
                category="partitioning",
                description=f"High file count ({total_files:,}) may indicate over-partitioning",
                recommendation="Reduce partition granularity or use Z-ORDER clustering",
                estimated_improvement="Faster file listing and query planning"
            ))

        # Issue: No partitioning for large datasets
        if total_files > 100 and len(partition_columns) == 0:
            issues.append(S3PerformanceIssue(
                severity="medium",
                category="partitioning",
                description="No partitioning detected for large dataset",
                recommendation="Partition by frequently filtered columns (e.g., date, region)",
                estimated_improvement="10-100x improvement for filtered queries"
            ))

        return issues

    def _generate_recommendations(
        self,
        issues: List[S3PerformanceIssue],
        file_formats: Dict[str, int],
        avg_size_mb: float
    ) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []

        # Priority recommendations based on issues
        high_severity = [i for i in issues if i.severity == "high"]
        if high_severity:
            recommendations.append("CRITICAL: Address high-severity issues first for maximum impact")

        # File size recommendations
        if avg_size_mb < self.OPTIMAL_MIN_FILE_SIZE_MB:
            recommendations.append(
                f"COMPACT FILES: Use Delta Lake OPTIMIZE or Spark coalesce() to achieve "
                f"128MB-1GB file sizes. Current average: {avg_size_mb:.1f}MB"
            )
            recommendations.append(
                "EXAMPLE: spark.read.parquet('path').coalesce(optimal_partitions).write.parquet('output')"
            )

        # Format recommendations
        if "csv" in file_formats or "json" in file_formats:
            recommendations.append(
                "CONVERT FORMAT: Migrate CSV/JSON to Parquet or Delta Lake:\n"
                "  - Parquet: 2-10x faster queries, columnar compression\n"
                "  - Delta Lake: ACID transactions, time travel, OPTIMIZE"
            )

        # Partitioning recommendations
        recommendations.append(
            "PARTITIONING STRATEGY:\n"
            "  - Partition by date for time-series data\n"
            "  - Avoid high-cardinality partition keys\n"
            "  - Target 100-10000 files per partition"
        )

        # S3 configuration recommendations
        recommendations.append(
            "S3 CONFIGURATION:\n"
            "  - Enable S3A committer: spark.hadoop.fs.s3a.committer.name=magic\n"
            "  - Use fast upload: spark.hadoop.fs.s3a.fast.upload=true\n"
            "  - Increase connections: spark.hadoop.fs.s3a.connection.maximum=100"
        )

        # Delta Lake recommendation
        recommendations.append(
            "CONSIDER DELTA LAKE:\n"
            "  - Automatic file compaction with OPTIMIZE\n"
            "  - Z-ORDER for multi-dimensional clustering\n"
            "  - VACUUM for storage cleanup\n"
            "  - Time travel for data versioning"
        )

        return recommendations

    def _calculate_score(
        self,
        issues: List[S3PerformanceIssue],
        avg_size_mb: float,
        file_formats: Dict[str, int]
    ) -> int:
        """Calculate overall performance score (0-100)."""
        score = 100

        # Deduct for issues
        for issue in issues:
            if issue.severity == "high":
                score -= 20
            elif issue.severity == "medium":
                score -= 10
            elif issue.severity == "low":
                score -= 5

        # Bonus for optimal file size
        if self.OPTIMAL_MIN_FILE_SIZE_MB <= avg_size_mb <= self.OPTIMAL_MAX_FILE_SIZE_MB:
            score += 10

        # Bonus for columnar format
        total_files = sum(file_formats.values())
        columnar = file_formats.get("parquet", 0) + file_formats.get("orc", 0)
        if total_files > 0 and columnar / total_files > 0.8:
            score += 10

        return max(0, min(100, score))

    # =========================================================================
    # OPTIMIZATION ACTIONS
    # =========================================================================

    def compact_files(
        self,
        source_path: str,
        target_path: str,
        target_file_size_mb: int = 256,
        file_format: str = "parquet",
        compression: str = "snappy"
    ) -> Dict[str, Any]:
        """
        Compact small files into optimally-sized files.

        Args:
            source_path: Source S3 path
            target_path: Target S3 path
            target_file_size_mb: Target file size in MB
            file_format: Output format (parquet, delta, orc)
            compression: Compression codec

        Returns:
            Dictionary with compaction results

        Example:
            >>> result = optimizer.compact_files(
            ...     "s3://bucket/raw/",
            ...     "s3://bucket/compacted/",
            ...     target_file_size_mb=256
            ... )
            >>> print(f"Reduced from {result['input_files']} to {result['output_files']} files")
        """
        spark = self._get_spark()

        # Read source data
        df = spark.read.format(file_format).load(source_path)

        # Calculate optimal partition count
        total_size = df.rdd.map(lambda x: len(str(x))).sum()
        target_size_bytes = target_file_size_mb * 1024 * 1024
        num_partitions = max(1, int(total_size / target_size_bytes))

        # Repartition and write
        df_compacted = df.coalesce(num_partitions)

        writer = df_compacted.write.format(file_format)
        if compression:
            writer = writer.option("compression", compression)

        writer.mode("overwrite").save(target_path)

        return {
            "source_path": source_path,
            "target_path": target_path,
            "input_files": df.rdd.getNumPartitions(),
            "output_files": num_partitions,
            "format": file_format,
            "compression": compression,
        }

    def convert_format(
        self,
        source_path: str,
        target_path: str,
        source_format: str,
        target_format: str = "parquet",
        compression: str = "snappy",
        partition_by: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Convert data to a more efficient format.

        Args:
            source_path: Source S3 path
            target_path: Target S3 path
            source_format: Source format (csv, json, etc.)
            target_format: Target format (parquet, delta, orc)
            compression: Compression codec
            partition_by: Partition columns

        Returns:
            Dictionary with conversion results

        Example:
            >>> result = optimizer.convert_format(
            ...     "s3://bucket/csv-data/",
            ...     "s3://bucket/parquet-data/",
            ...     source_format="csv",
            ...     target_format="parquet",
            ...     partition_by=["date"]
            ... )
        """
        spark = self._get_spark()

        # Read source data
        reader = spark.read.format(source_format)
        if source_format == "csv":
            reader = reader.option("header", "true").option("inferSchema", "true")
        df = reader.load(source_path)

        # Write in target format
        writer = df.write.format(target_format)
        if compression:
            writer = writer.option("compression", compression)
        if partition_by:
            writer = writer.partitionBy(*partition_by)

        writer.mode("overwrite").save(target_path)

        return {
            "source_path": source_path,
            "target_path": target_path,
            "source_format": source_format,
            "target_format": target_format,
            "rows_converted": df.count(),
            "partitioned_by": partition_by,
        }

    def get_storage_class_recommendation(
        self,
        access_frequency: str,
        data_criticality: str
    ) -> S3StorageClass:
        """
        Recommend S3 storage class based on access patterns.

        Args:
            access_frequency: "frequent", "infrequent", "rare", "archive"
            data_criticality: "critical", "important", "low"

        Returns:
            Recommended S3StorageClass

        Example:
            >>> storage_class = optimizer.get_storage_class_recommendation(
            ...     access_frequency="infrequent",
            ...     data_criticality="important"
            ... )
            >>> print(storage_class.value)
            'STANDARD_IA'
        """
        if access_frequency == "frequent":
            return S3StorageClass.STANDARD
        elif access_frequency == "infrequent":
            if data_criticality == "critical":
                return S3StorageClass.STANDARD_IA
            else:
                return S3StorageClass.ONEZONE_IA
        elif access_frequency == "rare":
            return S3StorageClass.GLACIER_INSTANT
        else:  # archive
            if data_criticality == "critical":
                return S3StorageClass.GLACIER
            else:
                return S3StorageClass.DEEP_ARCHIVE

    def estimate_cost_savings(
        self,
        current_size_gb: float,
        current_format: str,
        target_format: str = "parquet"
    ) -> Dict[str, float]:
        """
        Estimate cost savings from format conversion.

        Args:
            current_size_gb: Current data size in GB
            current_format: Current format (csv, json)
            target_format: Target format (parquet)

        Returns:
            Dictionary with estimated savings
        """
        # Compression ratios (approximate)
        compression_ratios = {
            "csv_to_parquet": 0.25,  # Parquet is ~75% smaller
            "json_to_parquet": 0.20,  # Parquet is ~80% smaller
        }

        ratio_key = f"{current_format}_to_{target_format}"
        ratio = compression_ratios.get(ratio_key, 0.5)

        new_size_gb = current_size_gb * ratio
        saved_gb = current_size_gb - new_size_gb

        # S3 pricing (approximate, us-east-1)
        s3_cost_per_gb = 0.023  # per month

        return {
            "current_size_gb": current_size_gb,
            "estimated_new_size_gb": new_size_gb,
            "storage_saved_gb": saved_gb,
            "compression_ratio": ratio,
            "monthly_savings_usd": saved_gb * s3_cost_per_gb,
            "yearly_savings_usd": saved_gb * s3_cost_per_gb * 12,
        }
