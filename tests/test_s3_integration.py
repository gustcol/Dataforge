"""
DataForge S3 Integration Tests

Tests for S3 storage optimization using moto (mock) and gofakes3 (fake S3 server).

Test Modes:
    - Mock mode (default): Uses moto for fast, isolated tests
    - Fake S3 mode: Uses gofakes3 for realistic S3 behavior testing

Running Tests:
    # Quick tests with moto
    pytest tests/test_s3_integration.py -v

    # Tests with gofakes3 (start server first)
    gofakes3 -backend memory -autobucket &
    S3_ENDPOINT=http://localhost:9000 pytest tests/test_s3_integration.py -v

Author: DataForge Team
"""

import os
import sys
import json
import tempfile
from pathlib import Path
from typing import Generator, Optional

import pytest
import boto3
from botocore.config import Config as BotoConfig

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================================
# FIXTURES
# ============================================================================

def get_s3_endpoint() -> Optional[str]:
    """Get S3 endpoint from environment (for gofakes3)."""
    return os.environ.get("S3_ENDPOINT")


def is_using_gofakes3() -> bool:
    """Check if tests should use gofakes3."""
    return get_s3_endpoint() is not None


@pytest.fixture(scope="module")
def s3_client():
    """Create S3 client - uses moto mock or gofakes3 based on environment."""
    endpoint = get_s3_endpoint()

    if endpoint:
        # Using gofakes3
        client = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id="test",
            aws_secret_access_key="test",
            region_name="us-east-1",
            config=BotoConfig(signature_version="s3v4"),
        )
        yield client
    else:
        # Using moto mock
        from moto import mock_aws

        with mock_aws():
            client = boto3.client(
                "s3",
                region_name="us-east-1",
                aws_access_key_id="test",
                aws_secret_access_key="test",
            )
            yield client


@pytest.fixture(scope="module")
def test_bucket(s3_client) -> Generator[str, None, None]:
    """Create a test bucket."""
    bucket_name = "dataforge-test-bucket"

    try:
        s3_client.create_bucket(Bucket=bucket_name)
    except s3_client.exceptions.BucketAlreadyOwnedByYou:
        pass
    except Exception:
        pass  # gofakes3 with -autobucket handles this

    yield bucket_name

    # Cleanup (best effort)
    try:
        # Delete all objects first
        response = s3_client.list_objects_v2(Bucket=bucket_name)
        if "Contents" in response:
            for obj in response["Contents"]:
                s3_client.delete_object(Bucket=bucket_name, Key=obj["Key"])
        s3_client.delete_bucket(Bucket=bucket_name)
    except Exception:
        pass


# ============================================================================
# S3 CONFIG TESTS
# ============================================================================

class TestS3Config:
    """Test S3 configuration classes."""

    def test_s3_config_defaults(self):
        """Test S3Config default values."""
        from dataforge.storage.s3_optimizer import S3Config

        config = S3Config()

        assert config.max_connections == 100
        assert config.ssl_enabled is True
        assert config.retry_max_attempts == 10
        assert config.multipart_threshold == 64 * 1024 * 1024  # 64MB in bytes
        assert config.multipart_chunksize == 64 * 1024 * 1024  # 64MB in bytes

    def test_s3_config_custom_values(self):
        """Test S3Config with custom values."""
        from dataforge.storage.s3_optimizer import S3Config

        config = S3Config(
            max_connections=200,
            ssl_enabled=False,
            retry_max_attempts=5,
            multipart_threshold=100 * 1024 * 1024,  # 100MB
        )

        assert config.max_connections == 200
        assert config.ssl_enabled is False
        assert config.retry_max_attempts == 5
        assert config.multipart_threshold == 100 * 1024 * 1024


# ============================================================================
# S3 OPTIMIZER TESTS
# ============================================================================

class TestS3Optimizer:
    """Test S3 optimizer functionality."""

    def test_optimizer_initialization(self):
        """Test S3Optimizer can be initialized."""
        from dataforge.storage.s3_optimizer import S3Optimizer, S3Config

        config = S3Config()
        optimizer = S3Optimizer(config=config)

        assert optimizer is not None
        assert optimizer.config == config

    def test_optimizer_initialization_no_spark(self):
        """Test S3Optimizer can be initialized without Spark."""
        from dataforge.storage.s3_optimizer import S3Optimizer, S3Config

        config = S3Config(max_connections=150)
        optimizer = S3Optimizer(config=config)

        # Should work without Spark for config-based methods
        assert optimizer.config.max_connections == 150

    def test_estimate_cost_savings(self):
        """Test cost savings estimation."""
        from dataforge.storage.s3_optimizer import S3Optimizer, S3Config

        optimizer = S3Optimizer(config=S3Config())

        savings = optimizer.estimate_cost_savings(
            current_size_gb=1000,
            current_format="csv",
            target_format="parquet",
        )

        assert isinstance(savings, dict)
        assert "storage_saved_gb" in savings
        assert savings["storage_saved_gb"] > 0
        assert "monthly_savings_usd" in savings

    def test_estimate_cost_savings_json(self):
        """Test cost savings for JSON conversion."""
        from dataforge.storage.s3_optimizer import S3Optimizer, S3Config

        optimizer = S3Optimizer(config=S3Config())

        savings = optimizer.estimate_cost_savings(
            current_size_gb=500,
            current_format="json",
            target_format="parquet",
        )

        assert savings["compression_ratio"] == 0.20
        assert savings["estimated_new_size_gb"] == 100  # 500 * 0.20

    def test_storage_class_recommendation(self):
        """Test storage class recommendations."""
        from dataforge.storage.s3_optimizer import S3Optimizer, S3Config, S3StorageClass

        optimizer = S3Optimizer(config=S3Config())

        # Frequently accessed data
        rec = optimizer.get_storage_class_recommendation(
            access_frequency="frequent",
            data_criticality="critical",
        )
        assert rec == S3StorageClass.STANDARD

        # Rarely accessed data
        rec = optimizer.get_storage_class_recommendation(
            access_frequency="rare",
            data_criticality="low",
        )
        assert rec == S3StorageClass.GLACIER_INSTANT

        # Archive data
        rec = optimizer.get_storage_class_recommendation(
            access_frequency="archive",
            data_criticality="critical",
        )
        assert rec == S3StorageClass.GLACIER


# ============================================================================
# S3 INTEGRATION TESTS (with real/fake S3)
# ============================================================================

class TestS3Integration:
    """Integration tests with S3 (moto or gofakes3)."""

    def test_upload_and_download(self, s3_client, test_bucket):
        """Test basic S3 upload and download."""
        key = "test-file.txt"
        content = b"Hello, DataForge S3 Testing!"

        # Upload
        s3_client.put_object(Bucket=test_bucket, Key=key, Body=content)

        # Download
        response = s3_client.get_object(Bucket=test_bucket, Key=key)
        downloaded = response["Body"].read()

        assert downloaded == content

    def test_list_objects(self, s3_client, test_bucket):
        """Test listing S3 objects."""
        # Create test files
        for i in range(5):
            s3_client.put_object(
                Bucket=test_bucket,
                Key=f"data/file_{i}.csv",
                Body=f"data{i}".encode(),
            )

        # List objects
        response = s3_client.list_objects_v2(
            Bucket=test_bucket,
            Prefix="data/",
        )

        assert "Contents" in response
        assert len(response["Contents"]) >= 5

    def test_json_data_storage(self, s3_client, test_bucket):
        """Test storing JSON data."""
        key = "config/settings.json"
        data = {
            "engine": "spark",
            "partitions": 200,
            "optimizations": ["adaptive", "broadcast"],
        }

        # Upload JSON
        s3_client.put_object(
            Bucket=test_bucket,
            Key=key,
            Body=json.dumps(data),
            ContentType="application/json",
        )

        # Download and verify
        response = s3_client.get_object(Bucket=test_bucket, Key=key)
        downloaded = json.loads(response["Body"].read().decode())

        assert downloaded == data

    def test_multipart_threshold_config(self, s3_client, test_bucket):
        """Test multipart upload configuration."""
        from dataforge.storage.s3_optimizer import S3Config

        config = S3Config(
            multipart_threshold=8 * 1024 * 1024,  # 8MB
            multipart_chunksize=8 * 1024 * 1024,  # 8MB
        )

        # Verify config values (in bytes)
        assert config.multipart_threshold == 8 * 1024 * 1024
        assert config.multipart_chunksize == 8 * 1024 * 1024

    def test_prefix_organization(self, s3_client, test_bucket):
        """Test S3 prefix organization patterns."""
        # Create hierarchical structure
        prefixes = [
            "raw/2024/01/data.csv",
            "processed/2024/01/data.parquet",
            "curated/users/data.delta",
        ]

        for prefix in prefixes:
            s3_client.put_object(
                Bucket=test_bucket,
                Key=prefix,
                Body=b"test",
            )

        # Verify structure
        for layer in ["raw", "processed", "curated"]:
            response = s3_client.list_objects_v2(
                Bucket=test_bucket,
                Prefix=f"{layer}/",
            )
            assert "Contents" in response


# ============================================================================
# FORMAT ADVISOR TESTS
# ============================================================================

class TestFormatAdvisor:
    """Test format recommendation functionality."""

    def test_recommend_parquet_for_analytics(self):
        """Test Parquet recommendation for analytics workloads."""
        from dataforge.storage.format_advisor import (
            FormatAdvisor,
            UseCase,
            FileFormat,
        )

        advisor = FormatAdvisor()
        rec = advisor.recommend(
            use_case=UseCase.ANALYTICS,
            data_size_gb=100,
        )

        assert rec.format in [FileFormat.PARQUET, FileFormat.DELTA]
        assert rec.score > 50

    def test_recommend_delta_for_acid(self):
        """Test Delta recommendation when ACID is required."""
        from dataforge.storage.format_advisor import (
            FormatAdvisor,
            UseCase,
            FileFormat,
        )

        advisor = FormatAdvisor()
        rec = advisor.recommend(
            use_case=UseCase.DATA_LAKE,
            data_size_gb=100,
            need_acid=True,
            need_time_travel=True,
        )

        assert rec.format == FileFormat.DELTA

    def test_format_comparison(self):
        """Test comparing different formats."""
        from dataforge.storage.format_advisor import FormatAdvisor, FileFormat

        advisor = FormatAdvisor()

        # Compare formats
        comparison = advisor.compare_formats([
            FileFormat.PARQUET,
            FileFormat.DELTA,
            FileFormat.ORC,
        ])

        assert isinstance(comparison, str)
        assert "Format Comparison" in comparison
        assert "parquet" in comparison.lower()
        assert "delta" in comparison.lower()


# ============================================================================
# STORAGE ANALYZER TESTS
# ============================================================================

class TestStorageAnalyzer:
    """Test storage analysis functionality."""

    def test_analyzer_initialization(self):
        """Test StorageAnalyzer initialization."""
        from dataforge.storage.storage_analyzer import StorageAnalyzer

        analyzer = StorageAnalyzer()
        assert analyzer is not None
        assert analyzer.spark is None  # Should start without Spark

    def test_storage_report_structure(self):
        """Test StorageReport dataclass structure."""
        from dataforge.storage.storage_analyzer import StorageReport

        report = StorageReport(
            path="s3://test/",
            total_size_bytes=1000000,
            total_files=10,
            file_size_distribution={"< 1MB": 5, "1-10MB": 5},
            format_distribution={"parquet": 8, "csv": 2},
            partition_analysis={},
            recommendations=["Consider compacting small files"]
        )

        assert report.total_files == 10
        assert report.total_size_bytes == 1000000
        assert len(report.recommendations) == 1

    def test_file_stats_structure(self):
        """Test FileStats dataclass structure."""
        from dataforge.storage.storage_analyzer import FileStats

        stats = FileStats(
            path="s3://bucket/file.parquet",
            size_bytes=1024 * 1024,
            format="parquet",
            compression="snappy",
            partition_values={"date": "2024-01-01"}
        )

        assert stats.size_bytes == 1024 * 1024
        assert stats.format == "parquet"
        assert stats.partition_values["date"] == "2024-01-01"


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # Check if running with gofakes3
    if is_using_gofakes3():
        print(f"Running tests with gofakes3 at: {get_s3_endpoint()}")
    else:
        print("Running tests with moto mock")

    pytest.main([__file__, "-v"])
