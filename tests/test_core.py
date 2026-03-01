"""
DataForge Core Functionality Tests

Test core functionality without external dependencies.

Run with:
    python -m pytest tests/test_core.py -v

Author: DataForge Team
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestEngineType:
    """Test EngineType enum."""

    def test_engine_values(self):
        """Test engine type values."""
        from dataforge.core.base import EngineType

        assert EngineType.PANDAS.value == "pandas"
        assert EngineType.POLARS.value == "polars"
        assert EngineType.SPARK.value == "spark"
        assert EngineType.RAPIDS.value == "rapids"
        assert EngineType.AUTO.value == "auto"

    def test_engine_from_string(self):
        """Test creating engine from string."""
        from dataforge.core.base import EngineType

        assert EngineType("pandas") == EngineType.PANDAS
        assert EngineType("spark") == EngineType.SPARK


class TestConfig:
    """Test configuration classes."""

    def test_pandas_config_defaults(self):
        """Test PandasConfig default values."""
        from dataforge.core.config import PandasConfig

        config = PandasConfig()
        assert config.max_memory_mb == 4096
        assert config.enable_copy_on_write is True
        assert config.optimize_dtypes is True

    def test_spark_config_defaults(self):
        """Test SparkConfig default values."""
        from dataforge.core.config import SparkConfig

        config = SparkConfig()
        assert config.shuffle_partitions == 200
        assert config.adaptive_enabled is True
        assert config.broadcast_threshold_mb == 10

    def test_rapids_config_defaults(self):
        """Test RapidsConfig default values."""
        from dataforge.core.config import RapidsConfig

        config = RapidsConfig()
        assert config.gpu_memory_fraction == 0.8
        assert config.fallback_to_pandas is True
        assert config.enable_spilling is True

    def test_polars_config_defaults(self):
        """Test PolarsConfig default values."""
        from dataforge.core.config import PolarsConfig

        config = PolarsConfig()
        assert config.use_lazy is True
        assert config.streaming is False
        assert config.rechunk is True
        assert config.parallel is True
        assert config.max_threads is None

    def test_dataforge_config(self):
        """Test DataForgeConfig."""
        from dataforge.core.config import DataForgeConfig
        from dataforge.core.base import EngineType

        config = DataForgeConfig()
        assert config.default_engine == EngineType.PANDAS
        assert config.auto_engine_selection is True


class TestExceptions:
    """Test custom exceptions."""

    def test_base_exception(self):
        """Test base exception."""
        from dataforge.core.exceptions import DataForgeError

        error = DataForgeError("Test error")
        assert str(error) == "Test error"

    def test_engine_not_available(self):
        """Test EngineNotAvailableError."""
        from dataforge.core.exceptions import EngineNotAvailableError

        error = EngineNotAvailableError("RAPIDS not installed")
        assert "RAPIDS" in str(error)

    def test_data_size_exceeded(self):
        """Test DataSizeExceededError."""
        from dataforge.core.exceptions import DataSizeExceededError

        error = DataSizeExceededError(
            actual_size_bytes=5_000_000_000,
            max_size_bytes=1_000_000_000,
            recommended_engine="spark"
        )
        assert "exceeds" in str(error).lower() or "GB" in str(error)


class TestEngineRecommender:
    """Test engine recommendation logic."""

    def test_recommend_pandas_for_small_data(self):
        """Test pandas recommendation for small data."""
        from dataforge.advisor.engine_recommender import EngineRecommender
        from dataforge.core.base import EngineType

        recommender = EngineRecommender()
        rec = recommender.recommend(data_size_mb=50)  # 50 MB

        assert rec.engine == EngineType.PANDAS

    def test_recommend_spark_for_large_data(self):
        """Test spark recommendation for large data."""
        from dataforge.advisor.engine_recommender import EngineRecommender
        from dataforge.core.base import EngineType

        recommender = EngineRecommender()
        rec = recommender.recommend(data_size_mb=20000, has_gpu=False)  # 20 GB

        assert rec.engine == EngineType.SPARK

    def test_recommend_polars_for_medium_data(self):
        """Test Polars recommendation for medium data without GPU."""
        from dataforge.advisor.engine_recommender import EngineRecommender
        from dataforge.core.base import EngineType

        recommender = EngineRecommender()
        rec = recommender.recommend(data_size_mb=500, has_gpu=False)  # 500 MB

        assert rec.engine == EngineType.POLARS

    def test_recommendation_has_reason(self):
        """Test that recommendation includes reason."""
        from dataforge.advisor.engine_recommender import EngineRecommender

        recommender = EngineRecommender()
        rec = recommender.recommend(data_size_mb=1000)  # 1 GB

        assert rec.reason is not None
        assert len(rec.reason) > 0


class TestHardwareDetector:
    """Test hardware detection."""

    def test_detect_hardware(self):
        """Test hardware detection."""
        from dataforge.advisor.hardware_detector import HardwareDetector

        detector = HardwareDetector()
        info = detector.detect_all()

        assert info.system_memory_gb > 0
        assert info.cpu_count > 0
        assert isinstance(info.gpu_available, bool)


class TestQualityValidators:
    """Test data quality validators."""

    def test_schema_validator(self):
        """Test schema validation."""
        from dataforge.quality.validators import SchemaValidator
        import pandas as pd

        df = pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["a", "b", "c"],
        })

        validator = SchemaValidator({
            "id": {"type": "int", "nullable": False},
            "name": {"type": "string", "nullable": True},
        })

        result = validator.validate(df)
        assert result.is_valid is True

    def test_column_validator_not_null(self):
        """Test column not null validation."""
        from dataforge.quality.validators import ColumnValidator
        import pandas as pd

        df = pd.DataFrame({
            "id": [1, 2, None],
        })

        validator = ColumnValidator(nullable=False)
        result = validator.validate(df["id"], column_name="id")

        assert result.is_valid is False

    def test_column_validator_range(self):
        """Test column range validation."""
        from dataforge.quality.validators import ColumnValidator
        import pandas as pd

        df = pd.DataFrame({
            "age": [25, 30, 150],  # 150 is out of range
        })

        validator = ColumnValidator(min=0, max=120)
        result = validator.validate(df["age"], column_name="age")

        assert result.is_valid is False


class TestDataProfiler:
    """Test data profiling."""

    def test_profile_dataframe(self):
        """Test DataFrame profiling."""
        from dataforge.quality.profiler import DataProfiler
        import pandas as pd

        df = pd.DataFrame({
            "id": [1, 2, 3, 4, 5],
            "name": ["a", "b", "c", "d", "e"],
            "value": [10.0, 20.0, 30.0, 40.0, 50.0],
        })

        profiler = DataProfiler()
        result = profiler.profile(df)

        assert result is not None
        assert result.column_count == 3
        assert result.row_count == 5


class TestFormatAdvisor:
    """Test format advisor."""

    def test_recommend_parquet_for_analytics(self):
        """Test Parquet recommendation for analytics."""
        from dataforge.storage.format_advisor import FormatAdvisor, UseCase, FileFormat

        advisor = FormatAdvisor()
        rec = advisor.recommend(
            use_case=UseCase.ANALYTICS,
            data_size_gb=100
        )

        # Should recommend Parquet or Delta for analytics
        assert rec.format in [FileFormat.PARQUET, FileFormat.DELTA]
        assert rec.score > 50

    def test_recommend_delta_with_acid(self):
        """Test Delta recommendation when ACID is required."""
        from dataforge.storage.format_advisor import FormatAdvisor, UseCase, FileFormat

        advisor = FormatAdvisor()
        rec = advisor.recommend(
            use_case=UseCase.DATA_LAKE,
            data_size_gb=100,
            need_acid=True,
            need_time_travel=True
        )

        assert rec.format == FileFormat.DELTA


class TestS3Config:
    """Test S3 configuration."""

    def test_s3_config_defaults(self):
        """Test S3Config default values."""
        from dataforge.storage.s3_optimizer import S3Config

        config = S3Config()
        assert config.max_connections == 100
        assert config.ssl_enabled is True
        assert config.retry_max_attempts == 10


class TestQualityChecks:
    """Test quality check functions."""

    def test_not_null_check(self):
        """Test not_null check."""
        from dataforge.quality.checks import QualityCheck
        import pandas as pd

        df = pd.DataFrame({"col": [1, 2, None]})
        check = QualityCheck.not_null("col")
        result = check.run(df)

        assert result.passed is False

    def test_unique_check(self):
        """Test unique check."""
        from dataforge.quality.checks import QualityCheck
        import pandas as pd

        df = pd.DataFrame({"col": [1, 2, 2]})  # Duplicate value
        check = QualityCheck.unique("col")
        result = check.run(df)

        assert result.passed is False

    def test_in_range_check(self):
        """Test in_range check."""
        from dataforge.quality.checks import QualityCheck
        import pandas as pd

        df = pd.DataFrame({"col": [1, 5, 10]})
        check = QualityCheck.in_range("col", min_val=0, max_val=100)
        result = check.run(df)

        assert result.passed is True


class TestBenchmarkProfiler:
    """Test benchmark profiler."""

    def test_profiler_context_manager(self):
        """Test profiler as context manager."""
        from dataforge.benchmarks.profiler import Profiler
        import time

        profiler = Profiler()

        with profiler.measure("test_operation"):
            time.sleep(0.01)  # 10ms

        result = profiler.get_timing("test_operation")
        assert result is not None
        assert result.duration_ms >= 10


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
