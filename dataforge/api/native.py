"""
DataForge Native Access Module

This module provides escape hatches to access underlying native DataFrames
and perform engine-specific operations not available in the unified API.

Use Cases:
    - Access Spark-specific functions (broadcast, checkpoint, etc.)
    - Use cuDF-specific GPU operations
    - Perform engine-specific optimizations
    - Integration with other libraries expecting native DataFrames

Example:
    >>> from dataforge import DataFrame
    >>> from dataforge.api.native import NativeAccess
    >>>
    >>> df = DataFrame.read_parquet("data.parquet", engine="spark")
    >>>
    >>> # Get native Spark DataFrame
    >>> spark_df = df.native
    >>>
    >>> # Use Spark-specific operations
    >>> from pyspark.sql.functions import broadcast
    >>> joined = spark_df.join(broadcast(small_df), "key")
"""

from typing import Any, Optional, TYPE_CHECKING

from dataforge.core.base import EngineType

if TYPE_CHECKING:
    import pandas as pd
    from pyspark.sql import DataFrame as SparkDataFrame
    import cudf


class NativeAccess:
    """
    Helper class for accessing native DataFrames.

    Provides type-checked access to underlying engine DataFrames
    and conversion utilities.

    Example:
        >>> native = NativeAccess(df)
        >>> if native.is_spark:
        ...     spark_df = native.as_spark()
        ...     spark_df.explain()
    """

    def __init__(self, dataframe: "DataFrame") -> None:
        """
        Initialize NativeAccess wrapper.

        Args:
            dataframe: DataForge DataFrame instance
        """
        from dataforge.api.unified import DataFrame
        self._df = dataframe

    @property
    def engine_type(self) -> EngineType:
        """Get the underlying engine type."""
        return self._df.engine_type

    @property
    def is_pandas(self) -> bool:
        """Check if underlying DataFrame is pandas."""
        return self.engine_type == EngineType.PANDAS

    @property
    def is_spark(self) -> bool:
        """Check if underlying DataFrame is Spark."""
        return self.engine_type == EngineType.SPARK

    @property
    def is_rapids(self) -> bool:
        """Check if underlying DataFrame is RAPIDS/cuDF."""
        return self.engine_type == EngineType.RAPIDS

    def as_pandas(self) -> "pd.DataFrame":
        """
        Get as pandas DataFrame.

        If not already pandas, converts (collects data to driver).

        Returns:
            pandas DataFrame

        Warning:
            Collects all data to memory if not already pandas.
        """
        return self._df.to_pandas()

    def as_spark(self) -> "SparkDataFrame":
        """
        Get as Spark DataFrame.

        If not already Spark, converts.

        Returns:
            Spark DataFrame
        """
        if self.is_spark:
            return self._df.native
        return self._df.to_spark().native

    def as_cudf(self) -> "cudf.DataFrame":
        """
        Get as cuDF DataFrame.

        If not already cuDF, converts.

        Returns:
            cuDF DataFrame
        """
        if self.is_rapids:
            return self._df.native
        return self._df.to_rapids().native

    @property
    def raw(self) -> Any:
        """
        Get the raw underlying DataFrame.

        Returns whatever the current engine uses internally.

        Returns:
            Native DataFrame (type depends on engine)
        """
        return self._df.native


def get_spark_session():
    """
    Get the current SparkSession.

    Utility function to access SparkSession from DataForge.

    Returns:
        SparkSession instance

    Example:
        >>> from dataforge.api.native import get_spark_session
        >>> spark = get_spark_session()
        >>> spark.sql("SHOW TABLES").show()
    """
    from dataforge.engines.spark_engine import SparkEngine
    engine = SparkEngine()
    return engine.spark


def get_spark_context():
    """
    Get the current SparkContext.

    Returns:
        SparkContext instance
    """
    return get_spark_session().sparkContext
