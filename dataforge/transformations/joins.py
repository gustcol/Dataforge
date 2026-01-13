"""
DataForge Join Transformations

Join utilities that work across all engines with best practices.
"""

from typing import Any, List, Optional, Union


def join_dataframes(
    left: Any,
    right: Any,
    on: Optional[Union[str, List[str]]] = None,
    left_on: Optional[Union[str, List[str]]] = None,
    right_on: Optional[Union[str, List[str]]] = None,
    how: str = "inner"
) -> Any:
    """
    Join two DataFrames.

    Args:
        left: Left DataFrame
        right: Right DataFrame
        on: Join columns (when same in both)
        left_on: Left join columns
        right_on: Right join columns
        how: Join type ('inner', 'left', 'right', 'outer')

    Returns:
        Joined DataFrame

    Best Practices:
        - Filter before joining to reduce data volume
        - Ensure join keys have same types
        - Use broadcast_join for small right tables
    """
    if hasattr(left, 'join') and hasattr(left, 'select'):
        # Spark DataFrame
        if on:
            return left.join(right, on=on, how=how)
        elif left_on and right_on:
            left_cols = [left_on] if isinstance(left_on, str) else left_on
            right_cols = [right_on] if isinstance(right_on, str) else right_on

            condition = None
            for l, r in zip(left_cols, right_cols):
                cond = left[l] == right[r]
                condition = cond if condition is None else condition & cond

            return left.join(right, on=condition, how=how)
    else:
        # Pandas/cuDF
        return left.merge(
            right,
            on=on,
            left_on=left_on,
            right_on=right_on,
            how=how
        )


def broadcast_join(
    left: Any,
    right: Any,
    on: Union[str, List[str]],
    how: str = "inner"
) -> Any:
    """
    Perform broadcast join (for Spark) or regular join (for others).

    Use when right table is small (< 10MB recommended).
    Avoids shuffle of left table for significant performance gain.

    Args:
        left: Left (large) DataFrame
        right: Right (small) DataFrame
        on: Join columns
        how: Join type

    Returns:
        Joined DataFrame
    """
    if hasattr(left, 'join') and hasattr(left, 'select'):
        # Spark - use broadcast
        from pyspark.sql.functions import broadcast
        return left.join(broadcast(right), on=on, how=how)
    else:
        # Pandas/cuDF - regular merge
        return left.merge(right, on=on, how=how)


def cross_join(left: Any, right: Any) -> Any:
    """
    Perform cross join (Cartesian product).

    Warning: Result size = left_rows * right_rows

    Args:
        left: Left DataFrame
        right: Right DataFrame

    Returns:
        Cross joined DataFrame
    """
    if hasattr(left, 'crossJoin'):
        # Spark
        return left.crossJoin(right)
    elif hasattr(left, 'merge'):
        # Pandas/cuDF
        left_copy = left.assign(__cross_key__=1)
        right_copy = right.assign(__cross_key__=1)
        result = left_copy.merge(right_copy, on="__cross_key__")
        return result.drop(columns=["__cross_key__"])
    else:
        raise TypeError(f"Unsupported DataFrame type: {type(left)}")
