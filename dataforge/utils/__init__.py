"""
DataForge Utils Module

Utility functions for logging, data conversion, and common operations.

Features:
    - Logging configuration
    - Data format converters
    - Helper functions

Example:
    >>> from dataforge.utils import setup_logging, convert_to_pandas
    >>>
    >>> # Configure logging
    >>> setup_logging(level="INFO")
    >>>
    >>> # Convert DataFrames
    >>> pandas_df = convert_to_pandas(spark_df)
"""

from dataforge.utils.logging import (
    setup_logging,
    get_logger,
    LogConfig,
)
from dataforge.utils.converters import (
    convert_to_pandas,
    convert_to_spark,
    convert_to_cudf,
    infer_engine_type,
    convert_schema,
)

__all__ = [
    # Logging
    "setup_logging",
    "get_logger",
    "LogConfig",
    # Converters
    "convert_to_pandas",
    "convert_to_spark",
    "convert_to_cudf",
    "infer_engine_type",
    "convert_schema",
]
