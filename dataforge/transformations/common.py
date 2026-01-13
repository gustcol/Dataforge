"""
DataForge Common Transformations

Basic transformation utilities that work across all engines.
"""

from typing import Any, Callable, Dict, List, Union


def filter_df(df: Any, condition: str) -> Any:
    """
    Filter DataFrame rows by condition.

    Works with pandas, Spark, and cuDF DataFrames.

    Args:
        df: Input DataFrame
        condition: SQL-like filter condition

    Returns:
        Filtered DataFrame
    """
    if hasattr(df, 'query'):
        return df.query(condition)
    elif hasattr(df, 'filter'):
        return df.filter(condition)
    else:
        raise TypeError(f"Unsupported DataFrame type: {type(df)}")


def select_columns(df: Any, columns: List[str]) -> Any:
    """
    Select specific columns from DataFrame.

    Args:
        df: Input DataFrame
        columns: List of column names

    Returns:
        DataFrame with selected columns
    """
    if hasattr(df, 'select'):
        return df.select(*columns)
    else:
        return df[columns]


def rename_columns(df: Any, mapping: Dict[str, str]) -> Any:
    """
    Rename columns in DataFrame.

    Args:
        df: Input DataFrame
        mapping: Old name to new name mapping

    Returns:
        DataFrame with renamed columns
    """
    if hasattr(df, 'withColumnRenamed'):
        result = df
        for old, new in mapping.items():
            result = result.withColumnRenamed(old, new)
        return result
    else:
        return df.rename(columns=mapping)


def add_column(
    df: Any,
    name: str,
    expression: Union[str, Callable]
) -> Any:
    """
    Add or replace a column.

    Args:
        df: Input DataFrame
        name: Column name
        expression: SQL expression or callable

    Returns:
        DataFrame with new column
    """
    if hasattr(df, 'withColumn'):
        from pyspark.sql import functions as F
        if isinstance(expression, str):
            return df.withColumn(name, F.expr(expression))
        else:
            return df.withColumn(name, expression(df))
    else:
        result = df.copy()
        if callable(expression):
            result[name] = expression(result)
        else:
            result[name] = result.eval(expression)
        return result


def drop_columns(df: Any, columns: List[str]) -> Any:
    """
    Drop columns from DataFrame.

    Args:
        df: Input DataFrame
        columns: Columns to drop

    Returns:
        DataFrame without dropped columns
    """
    if hasattr(df, 'drop') and hasattr(df, 'select'):
        return df.drop(*columns)
    else:
        return df.drop(columns=columns)
