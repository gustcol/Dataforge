"""
DataForge Advisor Module

This module provides intelligent engine selection based on:
    - Dataset size and characteristics
    - Available hardware (GPU, cluster)
    - Operation complexity
    - Performance requirements

Components:
    - EngineRecommender: Main recommendation engine
    - SizeAnalyzer: Dataset size analysis utilities
    - HardwareDetector: Hardware capability detection

The advisor uses a decision matrix to recommend the optimal engine:

    +----------------+------------+-----------+-------------------+
    | Data Size      | GPU        | Cluster   | Recommendation    |
    +----------------+------------+-----------+-------------------+
    | < 100 MB       | Any        | Any       | Pandas            |
    | 100MB - 1GB    | Available  | Any       | RAPIDS            |
    | 100MB - 1GB    | No         | No        | Pandas            |
    | 1GB - 10GB     | Available  | No        | RAPIDS            |
    | 1GB - 10GB     | No         | Available | Spark             |
    | > 10GB         | Any        | Available | Spark             |
    | > 10GB         | 32GB+ GPU  | No        | RAPIDS (chunked)  |
    +----------------+------------+-----------+-------------------+

Example:
    >>> from dataforge.advisor import EngineRecommender
    >>>
    >>> recommender = EngineRecommender()
    >>>
    >>> # Recommend based on file path
    >>> rec = recommender.recommend_for_path("large_data.parquet")
    >>> print(f"Use {rec.engine}: {rec.reason}")
    >>>
    >>> # Recommend based on known size
    >>> rec = recommender.recommend(data_size_mb=5000)
    >>> print(f"Use {rec.engine}: {rec.reason}")
"""

from dataforge.advisor.engine_recommender import (
    EngineRecommender,
    EngineRecommendation,
)
from dataforge.advisor.size_analyzer import SizeAnalyzer
from dataforge.advisor.hardware_detector import HardwareDetector

__all__ = [
    "EngineRecommender",
    "EngineRecommendation",
    "SizeAnalyzer",
    "HardwareDetector",
]
