"""
DataForge - Intelligent Data Processing Framework

A production-ready Python framework for data processing that intelligently
selects and utilizes the optimal processing engine (Pandas, PySpark, or
RAPIDS/cuDF) based on dataset characteristics, available hardware, and
use case requirements.

Features:
    - Unified API with engine-specific escape hatches
    - Full Databricks integration (Delta Lake, Unity Catalog, Photon)
    - Data quality validation and profiling
    - ML pipeline integration with MLflow
    - Structured Streaming support
    - Automatic engine selection based on data size and hardware

Example:
    >>> from dataforge import DataFrame, EngineRecommender
    >>>
    >>> # Get engine recommendation for your data
    >>> recommender = EngineRecommender()
    >>> recommendation = recommender.recommend(data_size_gb=5.0)
    >>> print(recommendation)
    EngineRecommendation(engine='spark', reason='Dataset > 1GB requires distributed processing')
    >>>
    >>> # Use the unified DataFrame API
    >>> df = DataFrame.read_csv("data.csv", engine="auto")
    >>> result = df.filter("age > 30").groupby("city").agg({"salary": "mean"})
    >>> result.write_parquet("output.parquet")

Author: DataForge Team
License: MIT
"""

__version__ = "1.0.0"
__author__ = "DataForge Team"

# Core components
from dataforge.core.config import (
    DataForgeConfig,
    EngineConfig,
    SparkConfig,
    RapidsConfig,
    PandasConfig,
    PolarsConfig,
)
from dataforge.core.exceptions import (
    DataForgeError,
    EngineNotAvailableError,
    DataSizeExceededError,
    ConfigurationError,
    ValidationError,
)
from dataforge.core.base import EngineType, DataFrameEngine

# Unified API
from dataforge.api.unified import DataFrame

# Advisor
from dataforge.advisor.engine_recommender import EngineRecommender, EngineRecommendation

# Engines (lazy imports for optional dependencies)
def get_pandas_engine():
    """Get the Pandas engine implementation."""
    from dataforge.engines.pandas_engine import PandasEngine
    return PandasEngine

def get_polars_engine():
    """Get the Polars engine implementation."""
    from dataforge.engines.polars_engine import PolarsEngine
    return PolarsEngine

def get_spark_engine():
    """Get the Spark engine implementation."""
    from dataforge.engines.spark_engine import SparkEngine
    return SparkEngine

def get_rapids_engine():
    """Get the RAPIDS engine implementation."""
    from dataforge.engines.rapids_engine import RapidsEngine
    return RapidsEngine

# Quality
from dataforge.quality.validators import SchemaValidator, ColumnValidator
from dataforge.quality.profiler import DataProfiler

# ML
from dataforge.ml.features import FeatureEngineer
from dataforge.ml.mlflow_utils import MLflowTracker

# Databricks
from dataforge.databricks.delta import DeltaTableManager
from dataforge.databricks.unity_catalog import UnityCatalogManager

# Benchmarks
from dataforge.benchmarks.profiler import Profiler
from dataforge.benchmarks.reporter import BenchmarkReporter

# Storage (lazy import for optional dependencies)
def get_s3_optimizer():
    """Get the S3 Optimizer for storage analysis."""
    from dataforge.storage.s3_optimizer import S3Optimizer
    return S3Optimizer

def get_storage_analyzer():
    """Get the Storage Analyzer."""
    from dataforge.storage.storage_analyzer import StorageAnalyzer
    return StorageAnalyzer

def get_format_advisor():
    """Get the Format Advisor."""
    from dataforge.storage.format_advisor import FormatAdvisor
    return FormatAdvisor

__all__ = [
    # Version
    "__version__",
    "__author__",
    # Config
    "DataForgeConfig",
    "EngineConfig",
    "SparkConfig",
    "RapidsConfig",
    "PandasConfig",
    "PolarsConfig",
    # Exceptions
    "DataForgeError",
    "EngineNotAvailableError",
    "DataSizeExceededError",
    "ConfigurationError",
    "ValidationError",
    # Core
    "EngineType",
    "DataFrameEngine",
    # API
    "DataFrame",
    # Advisor
    "EngineRecommender",
    "EngineRecommendation",
    # Engine factories
    "get_pandas_engine",
    "get_polars_engine",
    "get_spark_engine",
    "get_rapids_engine",
    # Quality
    "SchemaValidator",
    "ColumnValidator",
    "DataProfiler",
    # ML
    "FeatureEngineer",
    "MLflowTracker",
    # Databricks
    "DeltaTableManager",
    "UnityCatalogManager",
    # Benchmarks
    "Profiler",
    "BenchmarkReporter",
    # Storage
    "get_s3_optimizer",
    "get_storage_analyzer",
    "get_format_advisor",
]
