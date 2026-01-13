"""
DataForge Core Module

This module contains the foundational components of the DataForge framework:
    - Abstract base classes for engine implementations
    - Configuration management
    - Custom exception hierarchy
    - Type definitions

The core module establishes the contract that all engine implementations must follow,
ensuring consistent behavior across Pandas, Spark, and RAPIDS backends.
"""

from dataforge.core.base import (
    EngineType,
    DataFrameEngine,
    ReadOptions,
    WriteOptions,
)
from dataforge.core.config import (
    DataForgeConfig,
    EngineConfig,
    SparkConfig,
    RapidsConfig,
    PandasConfig,
)
from dataforge.core.exceptions import (
    DataForgeError,
    EngineNotAvailableError,
    DataSizeExceededError,
    ConfigurationError,
    ValidationError,
    TransformationError,
)

__all__ = [
    # Base
    "EngineType",
    "DataFrameEngine",
    "ReadOptions",
    "WriteOptions",
    # Config
    "DataForgeConfig",
    "EngineConfig",
    "SparkConfig",
    "RapidsConfig",
    "PandasConfig",
    # Exceptions
    "DataForgeError",
    "EngineNotAvailableError",
    "DataSizeExceededError",
    "ConfigurationError",
    "ValidationError",
    "TransformationError",
]
