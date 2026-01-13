"""
DataForge Aggregation Transformations

Aggregation utilities that work across all engines.
"""

from typing import Any, Dict, List, Optional, Union


def groupby_agg(
    df: Any,
    group_cols: List[str],
    aggregations: Dict[str, Union[str, List[str]]]
) -> Any:
    """
    Group by columns and apply aggregations.

    Args:
        df: Input DataFrame
        group_cols: Columns to group by
        aggregations: Column to aggregation mapping

    Returns:
        Aggregated DataFrame

    Example:
        >>> result = groupby_agg(
        ...     df,
        ...     ["region"],
        ...     {"amount": ["sum", "avg"], "id": "count"}
        ... )
    """
    if hasattr(df, 'groupBy'):
        # Spark DataFrame
        from pyspark.sql import functions as F

        agg_funcs = {
            "sum": F.sum, "avg": F.avg, "mean": F.mean,
            "min": F.min, "max": F.max, "count": F.count,
            "first": F.first, "last": F.last,
            "std": F.stddev, "var": F.variance,
        }

        agg_exprs = []
        for col, aggs in aggregations.items():
            if isinstance(aggs, str):
                aggs = [aggs]
            for agg in aggs:
                func = agg_funcs.get(agg.lower())
                if func:
                    agg_exprs.append(func(col).alias(f"{col}_{agg}"))

        return df.groupBy(*group_cols).agg(*agg_exprs)
    else:
        # Pandas/cuDF DataFrame
        agg_dict = {}
        for col, aggs in aggregations.items():
            agg_dict[col] = aggs if isinstance(aggs, list) else aggs

        return df.groupby(group_cols, as_index=False).agg(agg_dict)


def aggregate(
    df: Any,
    aggregations: Dict[str, Union[str, List[str]]]
) -> Any:
    """
    Apply aggregations without grouping.

    Args:
        df: Input DataFrame
        aggregations: Column to aggregation mapping

    Returns:
        Single-row DataFrame with results
    """
    if hasattr(df, 'agg') and hasattr(df, 'groupBy'):
        # Spark
        from pyspark.sql import functions as F

        agg_funcs = {
            "sum": F.sum, "avg": F.avg, "min": F.min,
            "max": F.max, "count": F.count,
        }

        agg_exprs = []
        for col, aggs in aggregations.items():
            if isinstance(aggs, str):
                aggs = [aggs]
            for agg in aggs:
                func = agg_funcs.get(agg.lower())
                if func:
                    agg_exprs.append(func(col).alias(f"{col}_{agg}"))

        return df.agg(*agg_exprs)
    else:
        return df.agg(aggregations)


def window_function(
    df: Any,
    column: str,
    function: str,
    partition_by: Optional[List[str]] = None,
    order_by: Optional[List[str]] = None,
    result_column: Optional[str] = None
) -> Any:
    """
    Apply window function.

    Args:
        df: Input DataFrame
        column: Column to apply function to
        function: Window function ('row_number', 'rank', 'dense_rank', 'lag', 'lead')
        partition_by: Partition columns
        order_by: Order columns
        result_column: Name for result column

    Returns:
        DataFrame with window function result
    """
    result_col = result_column or f"{column}_{function}"

    if hasattr(df, 'withColumn'):
        # Spark
        from pyspark.sql import Window, functions as F

        window_spec = Window.partitionBy(*(partition_by or []))
        if order_by:
            window_spec = window_spec.orderBy(*order_by)

        window_funcs = {
            "row_number": F.row_number,
            "rank": F.rank,
            "dense_rank": F.dense_rank,
            "sum": lambda: F.sum(column),
            "avg": lambda: F.avg(column),
            "min": lambda: F.min(column),
            "max": lambda: F.max(column),
        }

        func = window_funcs.get(function.lower())
        if func:
            return df.withColumn(result_col, func().over(window_spec))

    # Pandas implementation
    if partition_by:
        grouped = df.groupby(partition_by)
    else:
        grouped = df

    if function.lower() == "row_number":
        df[result_col] = grouped.cumcount() + 1
    elif function.lower() == "rank":
        if order_by:
            df[result_col] = df.groupby(partition_by or [])[order_by[0]].rank(method='min')
    elif function.lower() in ["sum", "avg", "min", "max"]:
        df[result_col] = grouped[column].transform(function.lower())

    return df
