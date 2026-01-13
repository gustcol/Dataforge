"""
DataForge Import Tests

Verify that all modules can be imported correctly.

Run with:
    python -m pytest tests/test_imports.py -v

Author: DataForge Team
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestCoreImports:
    """Test core module imports."""

    def test_import_main_package(self):
        """Test main package import."""
        import dataforge
        assert hasattr(dataforge, '__version__')
        assert dataforge.__version__ == "1.0.0"

    def test_import_config(self):
        """Test config imports."""
        from dataforge.core.config import (
            DataForgeConfig,
            EngineConfig,
            SparkConfig,
            RapidsConfig,
            PandasConfig,
        )
        assert DataForgeConfig is not None
        assert EngineConfig is not None

    def test_import_exceptions(self):
        """Test exception imports."""
        from dataforge.core.exceptions import (
            DataForgeError,
            EngineNotAvailableError,
            DataSizeExceededError,
            ConfigurationError,
            ValidationError,
        )
        assert issubclass(EngineNotAvailableError, DataForgeError)

    def test_import_base(self):
        """Test base class imports."""
        from dataforge.core.base import EngineType, DataFrameEngine
        assert EngineType.PANDAS.value == "pandas"
        assert EngineType.SPARK.value == "spark"
        assert EngineType.RAPIDS.value == "rapids"


class TestEngineImports:
    """Test engine module imports."""

    def test_import_pandas_engine(self):
        """Test Pandas engine import."""
        from dataforge.engines.pandas_engine import PandasEngine
        assert PandasEngine is not None

    def test_import_spark_engine(self):
        """Test Spark engine import."""
        from dataforge.engines.spark_engine import SparkEngine
        assert SparkEngine is not None

    def test_import_rapids_engine(self):
        """Test RAPIDS engine import."""
        from dataforge.engines.rapids_engine import RapidsEngine
        assert RapidsEngine is not None


class TestAPIImports:
    """Test API module imports."""

    def test_import_unified_api(self):
        """Test unified API import."""
        from dataforge.api.unified import DataFrame
        assert DataFrame is not None

    def test_import_native_api(self):
        """Test native API import."""
        from dataforge.api.native import NativeAccess
        assert NativeAccess is not None


class TestAdvisorImports:
    """Test advisor module imports."""

    def test_import_engine_recommender(self):
        """Test engine recommender import."""
        from dataforge.advisor.engine_recommender import (
            EngineRecommender,
            EngineRecommendation,
        )
        assert EngineRecommender is not None

    def test_import_hardware_detector(self):
        """Test hardware detector import."""
        from dataforge.advisor.hardware_detector import HardwareDetector
        assert HardwareDetector is not None

    def test_import_size_analyzer(self):
        """Test size analyzer import."""
        from dataforge.advisor.size_analyzer import SizeAnalyzer
        assert SizeAnalyzer is not None


class TestQualityImports:
    """Test quality module imports."""

    def test_import_validators(self):
        """Test validators import."""
        from dataforge.quality.validators import (
            SchemaValidator,
            ColumnValidator,
        )
        assert SchemaValidator is not None
        assert ColumnValidator is not None

    def test_import_profiler(self):
        """Test profiler import."""
        from dataforge.quality.profiler import DataProfiler
        assert DataProfiler is not None

    def test_import_checks(self):
        """Test quality checks import."""
        from dataforge.quality.checks import (
            QualityCheck,
            CheckResult,
            run_quality_checks,
        )
        assert QualityCheck is not None
        assert CheckResult is not None
        assert callable(run_quality_checks)


class TestMLImports:
    """Test ML module imports."""

    def test_import_features(self):
        """Test feature engineering import."""
        from dataforge.ml.features import FeatureEngineer
        assert FeatureEngineer is not None

    def test_import_pipeline(self):
        """Test ML pipeline import."""
        from dataforge.ml.pipeline import MLPipeline
        assert MLPipeline is not None

    def test_import_mlflow_utils(self):
        """Test MLflow utils import."""
        from dataforge.ml.mlflow_utils import MLflowTracker
        assert MLflowTracker is not None


class TestStreamingImports:
    """Test streaming module imports."""

    def test_import_sources(self):
        """Test streaming sources import."""
        from dataforge.streaming.sources import (
            KafkaSource,
            FileSource,
            DeltaSource,
        )
        assert KafkaSource is not None

    def test_import_sinks(self):
        """Test streaming sinks import."""
        from dataforge.streaming.sinks import (
            DeltaSink,
            KafkaSink,
        )
        assert DeltaSink is not None

    def test_import_processors(self):
        """Test streaming processors import."""
        from dataforge.streaming.processors import StreamProcessor
        assert StreamProcessor is not None


class TestDatabricksImports:
    """Test Databricks module imports."""

    def test_import_delta(self):
        """Test Delta Lake import."""
        from dataforge.databricks.delta import DeltaTableManager
        assert DeltaTableManager is not None

    def test_import_unity_catalog(self):
        """Test Unity Catalog import."""
        from dataforge.databricks.unity_catalog import UnityCatalogManager
        assert UnityCatalogManager is not None

    def test_import_photon(self):
        """Test Photon import."""
        from dataforge.databricks.photon import PhotonAnalyzer
        assert PhotonAnalyzer is not None

    def test_import_context(self):
        """Test Databricks context import."""
        from dataforge.databricks.context import DatabricksContext
        assert DatabricksContext is not None


class TestStorageImports:
    """Test storage module imports."""

    def test_import_s3_optimizer(self):
        """Test S3 optimizer import."""
        from dataforge.storage.s3_optimizer import (
            S3Optimizer,
            S3Config,
            S3PerformanceReport,
        )
        assert S3Optimizer is not None
        assert S3Config is not None

    def test_import_storage_analyzer(self):
        """Test storage analyzer import."""
        from dataforge.storage.storage_analyzer import (
            StorageAnalyzer,
            StorageReport,
        )
        assert StorageAnalyzer is not None

    def test_import_format_advisor(self):
        """Test format advisor import."""
        from dataforge.storage.format_advisor import (
            FormatAdvisor,
            FormatRecommendation,
        )
        assert FormatAdvisor is not None


class TestBenchmarkImports:
    """Test benchmark module imports."""

    def test_import_profiler(self):
        """Test benchmark profiler import."""
        from dataforge.benchmarks.profiler import Profiler
        assert Profiler is not None

    def test_import_reporter(self):
        """Test benchmark reporter import."""
        from dataforge.benchmarks.reporter import BenchmarkReporter
        assert BenchmarkReporter is not None


class TestUtilsImports:
    """Test utils module imports."""

    def test_import_logging(self):
        """Test logging utils import."""
        from dataforge.utils.logging import (
            setup_logging,
            get_logger,
            LogConfig,
        )
        assert callable(setup_logging)
        assert callable(get_logger)

    def test_import_converters(self):
        """Test converters import."""
        from dataforge.utils.converters import (
            convert_to_pandas,
            convert_to_spark,
            infer_engine_type,
        )
        assert callable(convert_to_pandas)
        assert callable(infer_engine_type)


class TestTransformationsImports:
    """Test transformations module imports."""

    def test_import_common(self):
        """Test common transformations import."""
        from dataforge.transformations.common import (
            filter_df,
            select_columns,
            rename_columns,
            add_column,
            drop_columns,
        )
        assert callable(filter_df)
        assert callable(select_columns)

    def test_import_aggregations(self):
        """Test aggregations import."""
        from dataforge.transformations.aggregations import (
            groupby_agg,
            aggregate,
            window_function,
        )
        assert callable(groupby_agg)
        assert callable(aggregate)

    def test_import_joins(self):
        """Test joins import."""
        from dataforge.transformations.joins import (
            join_dataframes,
            broadcast_join,
            cross_join,
        )
        assert callable(join_dataframes)
        assert callable(broadcast_join)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
