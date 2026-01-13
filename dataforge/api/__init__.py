"""
DataForge Unified API Module

This module provides a unified DataFrame API that abstracts the underlying
engine implementation while providing escape hatches to native functionality.

Components:
    - DataFrame: Unified interface for data operations
    - NativeAccess: Access to underlying engine DataFrames

The Unified API allows you to write code once and run it on any backend:
    - Pandas for small data and development
    - Spark for large-scale distributed processing
    - RAPIDS for GPU-accelerated analytics

Example:
    >>> from dataforge import DataFrame
    >>>
    >>> # Automatic engine selection based on data size
    >>> df = DataFrame.read_parquet("data.parquet", engine="auto")
    >>>
    >>> # Operations work identically across all engines
    >>> result = (df
    ...     .filter("amount > 100")
    ...     .groupby(["region"], {"amount": "sum"})
    ...     .sort(["amount"], ascending=False)
    ... )
    >>>
    >>> # Access native DataFrame when needed
    >>> native_df = result.native
    >>> spark_df = result.to_spark()  # Convert to Spark if needed
"""

from dataforge.api.unified import DataFrame
from dataforge.api.native import NativeAccess

__all__ = [
    "DataFrame",
    "NativeAccess",
]
