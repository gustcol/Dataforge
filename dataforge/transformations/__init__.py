"""
DataForge Transformations Module

This module provides common data transformation utilities that work
consistently across all supported engines.

Components:
    - Common: Basic transformations (filter, select, rename)
    - Aggregations: Group by and aggregation operations
    - Joins: Various join operations

All transformations are designed to:
    - Work identically across Pandas, Spark, and RAPIDS
    - Apply engine-specific optimizations automatically
    - Provide consistent error handling

Example:
    >>> from dataforge.transformations import filter_df, groupby_agg
    >>>
    >>> # Works with any engine
    >>> filtered = filter_df(df, "amount > 100")
    >>> aggregated = groupby_agg(df, ["region"], {"amount": "sum"})
"""

from dataforge.transformations.common import (
    filter_df,
    select_columns,
    rename_columns,
    add_column,
    drop_columns,
)
from dataforge.transformations.aggregations import (
    groupby_agg,
    aggregate,
    window_function,
)
from dataforge.transformations.joins import (
    join_dataframes,
    broadcast_join,
    cross_join,
)

__all__ = [
    # Common
    "filter_df",
    "select_columns",
    "rename_columns",
    "add_column",
    "drop_columns",
    # Aggregations
    "groupby_agg",
    "aggregate",
    "window_function",
    # Joins
    "join_dataframes",
    "broadcast_join",
    "cross_join",
]
