"""
DataForge Unified DataFrame API

This module provides a unified DataFrame interface that works across all
supported engines (Pandas, Spark, RAPIDS) with consistent API and behavior.

The DataFrame class wraps the underlying engine implementation and provides:
    - Consistent method chaining API
    - Automatic engine selection
    - Lazy evaluation where supported
    - Seamless engine switching

Design Philosophy:
    - Write once, run anywhere
    - Escape hatches for engine-specific features
    - Best practices applied automatically
    - Clear error messages

Example:
    >>> from dataforge import DataFrame
    >>>
    >>> # Read with automatic engine selection
    >>> df = DataFrame.read_csv("large_data.csv", engine="auto")
    >>>
    >>> # Chain operations
    >>> result = (df
    ...     .filter("status == 'active'")
    ...     .select(["id", "name", "amount"])
    ...     .groupby(["region"], {"amount": ["sum", "avg"]})
    ...     .sort(["amount_sum"], ascending=False)
    ...     .limit(100)
    ... )
    >>>
    >>> # Write results
    >>> result.write_parquet("output/", partition_by=["region"])
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union, TYPE_CHECKING

from dataforge.core.base import (
    DataFrameEngine,
    EngineType,
    JoinType,
    ReadOptions,
    WriteOptions,
)
from dataforge.core.config import DataForgeConfig
from dataforge.core.exceptions import DataForgeError, EngineNotAvailableError

if TYPE_CHECKING:
    import pandas as pd


class DataFrame:
    """
    Unified DataFrame API for DataForge.

    This class provides a consistent interface for data operations across
    all supported engines. It wraps the underlying engine DataFrame and
    provides method chaining for fluent data transformations.

    Attributes:
        _data: The underlying native DataFrame
        _engine: The DataFrameEngine instance
        _config: DataForge configuration

    Class Methods:
        read_csv: Read CSV file(s)
        read_parquet: Read Parquet file(s)
        read_json: Read JSON file(s)
        read_delta: Read Delta Lake table
        from_pandas: Create from pandas DataFrame

    Instance Methods:
        filter: Filter rows by condition
        select: Select specific columns
        rename: Rename columns
        with_column: Add or replace column
        drop: Drop columns
        distinct: Remove duplicates
        sort: Sort by columns
        limit: Limit number of rows
        groupby: Group and aggregate
        join: Join with another DataFrame
        write_csv: Write to CSV
        write_parquet: Write to Parquet
        write_delta: Write to Delta Lake

    Properties:
        native: Access underlying DataFrame
        engine_type: Current engine type
        columns: List of column names
        dtypes: Column data types
        shape: (rows, columns) tuple

    Example:
        >>> df = DataFrame.read_parquet("data/", engine="spark")
        >>> result = df.filter("amount > 100").groupby(["region"], {"amount": "sum"})
        >>> result.write_delta("output/", mode="overwrite")
    """

    def __init__(
        self,
        data: Any,
        engine: DataFrameEngine,
        config: Optional[DataForgeConfig] = None
    ) -> None:
        """
        Initialize DataFrame wrapper.

        Args:
            data: Native DataFrame (pandas, spark, or cudf)
            engine: DataFrameEngine instance
            config: Optional configuration
        """
        self._data = data
        self._engine = engine
        self._config = config or DataForgeConfig.get_global()

    @classmethod
    def _get_engine(
        cls,
        engine: Union[str, EngineType],
        data_path: Optional[str] = None,
        config: Optional[DataForgeConfig] = None
    ) -> DataFrameEngine:
        """
        Get or create appropriate engine instance.

        Args:
            engine: Engine type or 'auto' for automatic selection
            data_path: Optional path for size-based selection
            config: Optional configuration

        Returns:
            Appropriate DataFrameEngine instance
        """
        config = config or DataForgeConfig.get_global()

        if isinstance(engine, str):
            engine = EngineType(engine.lower())

        if engine == EngineType.AUTO:
            # Use recommender for automatic selection
            from dataforge.advisor.engine_recommender import EngineRecommender

            recommender = EngineRecommender(config.engine_selection)

            if data_path and os.path.exists(data_path):
                recommendation = recommender.recommend_for_path(data_path)
            else:
                recommendation = recommender.recommend(data_size_mb=100)  # Default

            engine = recommendation.engine

        # Create engine instance
        if engine == EngineType.PANDAS:
            from dataforge.engines.pandas_engine import PandasEngine
            return PandasEngine(config.pandas)
        elif engine == EngineType.SPARK:
            from dataforge.engines.spark_engine import SparkEngine
            return SparkEngine(config.spark)
        elif engine == EngineType.RAPIDS:
            from dataforge.engines.rapids_engine import RapidsEngine
            return RapidsEngine(config.rapids)
        else:
            raise DataForgeError(f"Unknown engine type: {engine}")

    # ==========================================================================
    # CLASS METHODS - READ OPERATIONS
    # ==========================================================================

    @classmethod
    def read_csv(
        cls,
        path: Union[str, Path, List[str]],
        engine: Union[str, EngineType] = EngineType.AUTO,
        config: Optional[DataForgeConfig] = None,
        **kwargs
    ) -> "DataFrame":
        """
        Read CSV file(s) into a DataFrame.

        Args:
            path: Path to CSV file(s), supports glob patterns
            engine: Engine to use ('auto', 'pandas', 'spark', 'rapids')
            config: Optional configuration
            **kwargs: Additional options passed to ReadOptions

        Returns:
            DataFrame instance

        Example:
            >>> df = DataFrame.read_csv("data/*.csv", engine="auto")
            >>> df = DataFrame.read_csv("data.csv", columns=["id", "name"])
        """
        path_str = str(path) if not isinstance(path, list) else path[0]
        engine_instance = cls._get_engine(engine, path_str, config)
        options = ReadOptions(**kwargs) if kwargs else None
        data = engine_instance.read_csv(path, options)
        return cls(data, engine_instance, config)

    @classmethod
    def read_parquet(
        cls,
        path: Union[str, Path, List[str]],
        engine: Union[str, EngineType] = EngineType.AUTO,
        config: Optional[DataForgeConfig] = None,
        **kwargs
    ) -> "DataFrame":
        """
        Read Parquet file(s) into a DataFrame.

        Parquet is the recommended format for analytical workloads.

        Args:
            path: Path to Parquet file(s) or directory
            engine: Engine to use
            config: Optional configuration
            **kwargs: Additional options passed to ReadOptions

        Returns:
            DataFrame instance

        Example:
            >>> df = DataFrame.read_parquet("data/")
            >>> df = DataFrame.read_parquet("data.parquet", columns=["id", "amount"])
        """
        path_str = str(path) if not isinstance(path, list) else path[0]
        engine_instance = cls._get_engine(engine, path_str, config)
        options = ReadOptions(**kwargs) if kwargs else None
        data = engine_instance.read_parquet(path, options)
        return cls(data, engine_instance, config)

    @classmethod
    def read_json(
        cls,
        path: Union[str, Path, List[str]],
        engine: Union[str, EngineType] = EngineType.AUTO,
        config: Optional[DataForgeConfig] = None,
        **kwargs
    ) -> "DataFrame":
        """
        Read JSON file(s) into a DataFrame.

        Args:
            path: Path to JSON file(s)
            engine: Engine to use
            config: Optional configuration
            **kwargs: Additional options passed to ReadOptions

        Returns:
            DataFrame instance
        """
        path_str = str(path) if not isinstance(path, list) else path[0]
        engine_instance = cls._get_engine(engine, path_str, config)
        options = ReadOptions(**kwargs) if kwargs else None
        data = engine_instance.read_json(path, options)
        return cls(data, engine_instance, config)

    @classmethod
    def read_delta(
        cls,
        path: Union[str, Path],
        engine: Union[str, EngineType] = EngineType.AUTO,
        config: Optional[DataForgeConfig] = None,
        version: Optional[int] = None,
        timestamp: Optional[str] = None,
        **kwargs
    ) -> "DataFrame":
        """
        Read Delta Lake table into a DataFrame.

        Supports time travel via version or timestamp.

        Args:
            path: Path to Delta table
            engine: Engine to use
            config: Optional configuration
            version: Specific version to read
            timestamp: Timestamp for time travel
            **kwargs: Additional options

        Returns:
            DataFrame instance

        Example:
            >>> # Read latest version
            >>> df = DataFrame.read_delta("delta_table/")
            >>>
            >>> # Read specific version
            >>> df = DataFrame.read_delta("delta_table/", version=5)
        """
        engine_instance = cls._get_engine(engine, str(path), config)
        options = ReadOptions(**kwargs) if kwargs else None
        data = engine_instance.read_delta(path, options, version, timestamp)
        return cls(data, engine_instance, config)

    @classmethod
    def from_pandas(
        cls,
        df: "pd.DataFrame",
        engine: Union[str, EngineType] = EngineType.AUTO,
        config: Optional[DataForgeConfig] = None
    ) -> "DataFrame":
        """
        Create DataFrame from pandas DataFrame.

        Args:
            df: pandas DataFrame
            engine: Engine to convert to
            config: Optional configuration

        Returns:
            DataFrame instance
        """
        engine_instance = cls._get_engine(engine, None, config)
        data = engine_instance.from_pandas(df)
        return cls(data, engine_instance, config)

    # ==========================================================================
    # PROPERTIES
    # ==========================================================================

    @property
    def native(self) -> Any:
        """
        Access the underlying native DataFrame.

        Returns:
            Native DataFrame (pandas, Spark, or cuDF)

        Example:
            >>> spark_df = df.native  # Get Spark DataFrame
            >>> spark_df.createOrReplaceTempView("my_table")
        """
        return self._data

    @property
    def engine_type(self) -> EngineType:
        """Get the current engine type."""
        return self._engine.engine_type

    @property
    def columns(self) -> List[str]:
        """Get list of column names."""
        return self._engine.columns(self._data)

    @property
    def dtypes(self) -> Dict[str, str]:
        """Get column data types."""
        return self._engine.dtypes(self._data)

    @property
    def shape(self) -> tuple:
        """Get (rows, columns) shape."""
        return (self._engine.count(self._data), len(self.columns))

    # ==========================================================================
    # TRANSFORMATION METHODS
    # ==========================================================================

    def filter(self, condition: str) -> "DataFrame":
        """
        Filter rows by condition.

        Args:
            condition: SQL-like filter expression

        Returns:
            Filtered DataFrame

        Example:
            >>> df.filter("age > 18 and status == 'active'")
        """
        data = self._engine.filter(self._data, condition)
        return DataFrame(data, self._engine, self._config)

    def select(self, columns: List[str]) -> "DataFrame":
        """
        Select specific columns.

        Args:
            columns: List of column names

        Returns:
            DataFrame with selected columns

        Example:
            >>> df.select(["id", "name", "email"])
        """
        data = self._engine.select(self._data, columns)
        return DataFrame(data, self._engine, self._config)

    def rename(self, columns: Dict[str, str]) -> "DataFrame":
        """
        Rename columns.

        Args:
            columns: Mapping of old names to new names

        Returns:
            DataFrame with renamed columns

        Example:
            >>> df.rename({"old_name": "new_name"})
        """
        data = self._engine.rename(self._data, columns)
        return DataFrame(data, self._engine, self._config)

    def with_column(
        self,
        name: str,
        expression: Union[str, Callable]
    ) -> "DataFrame":
        """
        Add or replace a column.

        Args:
            name: Column name
            expression: SQL expression or callable

        Returns:
            DataFrame with new column

        Example:
            >>> df.with_column("total", "price * quantity")
            >>> df.with_column("upper_name", lambda d: d["name"].str.upper())
        """
        data = self._engine.with_column(self._data, name, expression)
        return DataFrame(data, self._engine, self._config)

    def drop(self, columns: Union[str, List[str]]) -> "DataFrame":
        """
        Drop columns.

        Args:
            columns: Column name(s) to drop

        Returns:
            DataFrame without dropped columns
        """
        if isinstance(columns, str):
            columns = [columns]
        data = self._engine.drop(self._data, columns)
        return DataFrame(data, self._engine, self._config)

    def distinct(self) -> "DataFrame":
        """
        Return distinct rows.

        Returns:
            DataFrame with duplicate rows removed
        """
        data = self._engine.distinct(self._data)
        return DataFrame(data, self._engine, self._config)

    def sort(
        self,
        columns: Union[str, List[str]],
        ascending: Union[bool, List[bool]] = True
    ) -> "DataFrame":
        """
        Sort by columns.

        Args:
            columns: Column(s) to sort by
            ascending: Sort direction(s)

        Returns:
            Sorted DataFrame
        """
        if isinstance(columns, str):
            columns = [columns]
        data = self._engine.sort(self._data, columns, ascending)
        return DataFrame(data, self._engine, self._config)

    def limit(self, n: int) -> "DataFrame":
        """
        Limit to first n rows.

        Args:
            n: Number of rows

        Returns:
            DataFrame with at most n rows
        """
        data = self._engine.limit(self._data, n)
        return DataFrame(data, self._engine, self._config)

    # ==========================================================================
    # AGGREGATION METHODS
    # ==========================================================================

    def groupby(
        self,
        columns: Union[str, List[str]],
        aggregations: Dict[str, Union[str, List[str]]]
    ) -> "DataFrame":
        """
        Group by columns and aggregate.

        Args:
            columns: Column(s) to group by
            aggregations: Column to aggregation mapping

        Returns:
            Aggregated DataFrame

        Example:
            >>> df.groupby(["region"], {"amount": ["sum", "avg"], "id": "count"})
        """
        if isinstance(columns, str):
            columns = [columns]
        data = self._engine.groupby(self._data, columns, aggregations)
        return DataFrame(data, self._engine, self._config)

    def agg(
        self,
        aggregations: Dict[str, Union[str, List[str]]]
    ) -> "DataFrame":
        """
        Aggregate without grouping.

        Args:
            aggregations: Column to aggregation mapping

        Returns:
            Single-row DataFrame with results
        """
        data = self._engine.agg(self._data, aggregations)
        return DataFrame(data, self._engine, self._config)

    # ==========================================================================
    # JOIN METHODS
    # ==========================================================================

    def join(
        self,
        other: "DataFrame",
        on: Optional[Union[str, List[str]]] = None,
        left_on: Optional[Union[str, List[str]]] = None,
        right_on: Optional[Union[str, List[str]]] = None,
        how: Union[str, JoinType] = JoinType.INNER
    ) -> "DataFrame":
        """
        Join with another DataFrame.

        Args:
            other: DataFrame to join with
            on: Join column(s) when same in both
            left_on: Left join column(s)
            right_on: Right join column(s)
            how: Join type ('inner', 'left', 'right', 'outer')

        Returns:
            Joined DataFrame

        Example:
            >>> orders.join(customers, on="customer_id", how="left")
        """
        if isinstance(how, str):
            how = JoinType(how.lower())

        data = self._engine.join(
            self._data,
            other.native,
            on=on,
            left_on=left_on,
            right_on=right_on,
            how=how
        )
        return DataFrame(data, self._engine, self._config)

    # ==========================================================================
    # WRITE METHODS
    # ==========================================================================

    def write_csv(
        self,
        path: Union[str, Path],
        mode: str = "error",
        **kwargs
    ) -> None:
        """
        Write to CSV file(s).

        Args:
            path: Output path
            mode: Write mode ('overwrite', 'append', 'error')
            **kwargs: Additional WriteOptions
        """
        options = WriteOptions(mode=mode, **kwargs)
        self._engine.write_csv(self._data, path, options)

    def write_parquet(
        self,
        path: Union[str, Path],
        mode: str = "error",
        partition_by: Optional[List[str]] = None,
        **kwargs
    ) -> None:
        """
        Write to Parquet file(s).

        Args:
            path: Output path
            mode: Write mode
            partition_by: Partition columns
            **kwargs: Additional WriteOptions
        """
        options = WriteOptions(mode=mode, partition_by=partition_by, **kwargs)
        self._engine.write_parquet(self._data, path, options)

    def write_delta(
        self,
        path: Union[str, Path],
        mode: str = "error",
        partition_by: Optional[List[str]] = None,
        optimize: bool = False,
        z_order_by: Optional[List[str]] = None,
        **kwargs
    ) -> None:
        """
        Write to Delta Lake table.

        Args:
            path: Output path
            mode: Write mode
            partition_by: Partition columns
            optimize: Run OPTIMIZE after write
            z_order_by: Columns for Z-ORDER
            **kwargs: Additional WriteOptions
        """
        options = WriteOptions(
            mode=mode,
            partition_by=partition_by,
            optimize=optimize,
            z_order_by=z_order_by,
            **kwargs
        )
        self._engine.write_delta(self._data, path, options)

    # ==========================================================================
    # CONVERSION METHODS
    # ==========================================================================

    def to_pandas(self) -> "pd.DataFrame":
        """
        Convert to pandas DataFrame.

        Warning: Collects all data to driver memory.

        Returns:
            pandas DataFrame
        """
        return self._engine.to_pandas(self._data)

    def to_spark(self) -> "DataFrame":
        """
        Convert to Spark DataFrame.

        Returns:
            DataFrame with Spark engine
        """
        if self.engine_type == EngineType.SPARK:
            return self

        from dataforge.engines.spark_engine import SparkEngine
        spark_engine = SparkEngine(self._config.spark if self._config else None)
        pandas_df = self.to_pandas()
        spark_df = spark_engine.from_pandas(pandas_df)
        return DataFrame(spark_df, spark_engine, self._config)

    def to_rapids(self) -> "DataFrame":
        """
        Convert to RAPIDS DataFrame.

        Returns:
            DataFrame with RAPIDS engine
        """
        if self.engine_type == EngineType.RAPIDS:
            return self

        from dataforge.engines.rapids_engine import RapidsEngine
        rapids_engine = RapidsEngine(self._config.rapids if self._config else None)
        pandas_df = self.to_pandas()
        cudf_df = rapids_engine.from_pandas(pandas_df)
        return DataFrame(cudf_df, rapids_engine, self._config)

    # ==========================================================================
    # UTILITY METHODS
    # ==========================================================================

    def count(self) -> int:
        """Count number of rows."""
        return self._engine.count(self._data)

    def cache(self) -> "DataFrame":
        """Cache DataFrame for reuse."""
        data = self._engine.cache(self._data)
        return DataFrame(data, self._engine, self._config)

    def uncache(self) -> "DataFrame":
        """Remove from cache."""
        data = self._engine.uncache(self._data)
        return DataFrame(data, self._engine, self._config)

    def show(self, n: int = 20, truncate: bool = True) -> None:
        """Display rows."""
        self._engine.show(self._data, n, truncate)

    def collect(self) -> List[Dict[str, Any]]:
        """Collect all rows as list of dictionaries."""
        return self._engine.collect(self._data)

    def head(self, n: int = 5) -> List[Dict[str, Any]]:
        """Get first n rows as dictionaries."""
        return self._engine.head(self._data, n)

    def schema(self) -> Dict[str, Any]:
        """Get schema information."""
        return self._engine.schema(self._data)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"DataFrame(engine={self.engine_type.value}, "
            f"columns={len(self.columns)}, "
            f"rows={self.count()})"
        )

    def __len__(self) -> int:
        """Return row count."""
        return self.count()
