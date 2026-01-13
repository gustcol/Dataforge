"""
DataForge Storage Module

Cloud storage optimization and analysis utilities.

Features:
    - S3 performance analysis and optimization
    - File format recommendations
    - Partitioning strategies
    - Storage cost optimization

Example:
    >>> from dataforge.storage import S3Optimizer, StorageAnalyzer
    >>>
    >>> # Analyze S3 path
    >>> analyzer = StorageAnalyzer(spark)
    >>> report = analyzer.analyze("s3://bucket/path/")
    >>>
    >>> # Get optimization recommendations
    >>> optimizer = S3Optimizer(spark)
    >>> recommendations = optimizer.get_recommendations("s3://bucket/path/")
"""

from dataforge.storage.s3_optimizer import (
    S3Optimizer,
    S3Config,
    S3PerformanceReport,
)
from dataforge.storage.storage_analyzer import (
    StorageAnalyzer,
    StorageReport,
    FileStats,
)
from dataforge.storage.format_advisor import (
    FormatAdvisor,
    FormatRecommendation,
)

__all__ = [
    # S3 Optimizer
    "S3Optimizer",
    "S3Config",
    "S3PerformanceReport",
    # Storage Analyzer
    "StorageAnalyzer",
    "StorageReport",
    "FileStats",
    # Format Advisor
    "FormatAdvisor",
    "FormatRecommendation",
]
