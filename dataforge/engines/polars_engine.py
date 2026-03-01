"""
DataForge Polars Engine Implementation

This module provides a Polars-based implementation of the DataFrameEngine interface.
Polars is a high-performance DataFrame library written in Rust that provides
lazy evaluation, parallel execution, and memory-efficient processing.

Performance Characteristics:
    - 5-20x faster than pandas for most operations
    - Lazy evaluation enables automatic query optimization
    - Parallel execution on all CPU cores by default
    - Streaming mode for datasets larger than memory
    - Native Rust backend avoids Python GIL limitations

Recommended Use Cases:
    - Datasets from 100MB to 10GB on a single node
    - Complex aggregation and join-heavy workloads
    - ETL pipelines requiring high throughput
    - Workloads where Spark overhead isn't justified

Example:
    >>> from dataforge.engines import PolarsEngine
    >>> from dataforge.core import ReadOptions
    >>>
    >>> engine = PolarsEngine()
    >>>
    >>> # Basic read
    >>> df = engine.read_csv("data.csv")
    >>>
    >>> # Optimized read with options
    >>> options = ReadOptions(columns=["id", "amount"])
    >>> df = engine.read_parquet("data.parquet", options)
"""

import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Union, cast

try:
    import polars as pl

    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False
    pl = None  # type: ignore[assignment]

from dataforge.core.base import (
    DataFrameEngine,
    EngineType,
    JoinType,
    ReadOptions,
    WriteOptions,
)
from dataforge.core.config import PolarsConfig
from dataforge.core.exceptions import (
    DataForgeError,
    EngineNotAvailableError,
    TransformationError,
)


# Map DataForge JoinType to Polars join strategy
_JOIN_MAP: Dict[JoinType, str] = {
    JoinType.INNER: "inner",
    JoinType.LEFT: "left",
    JoinType.RIGHT: "right",
    JoinType.OUTER: "full",
    JoinType.CROSS: "cross",
    JoinType.LEFT_SEMI: "semi",
    JoinType.LEFT_ANTI: "anti",
}


class PolarsEngine(DataFrameEngine["pl.DataFrame"]):
    """
    Polars implementation of DataFrameEngine.

    Provides high-performance DataFrame operations using the Polars library,
    which is written in Rust and supports lazy evaluation for query optimization.

    Attributes:
        config: PolarsConfig instance with engine settings
    """

    def __init__(self, config: Optional[PolarsConfig] = None) -> None:
        if not POLARS_AVAILABLE:
            raise EngineNotAvailableError(
                "Polars is not installed. Install with: pip install polars"
            )
        self.config = config or PolarsConfig()

    @staticmethod
    def check_availability() -> bool:
        """Check if Polars is available."""
        return POLARS_AVAILABLE

    @property
    def engine_type(self) -> EngineType:
        return EngineType.POLARS

    @property
    def is_available(self) -> bool:
        return POLARS_AVAILABLE

    # ==========================================================================
    # READ OPERATIONS
    # ==========================================================================

    def read_csv(
        self,
        path: Union[str, Path, List[str]],
        options: Optional[ReadOptions] = None,
    ) -> "pl.DataFrame":
        options = options or ReadOptions()
        try:
            kwargs: Dict[str, Any] = {
                "has_header": options.header,
                "separator": options.delimiter,
                "infer_schema_length": 10000 if options.infer_schema else 0,
                "null_values": options.null_values,
                "encoding": "utf8" if options.encoding == "utf-8" else options.encoding,
            }
            if options.columns:
                kwargs["columns"] = options.columns
            if options.sample_fraction is not None:
                # Read then sample for CSV
                df = pl.read_csv(str(path), **kwargs)
                return df.sample(fraction=options.sample_fraction)
            return pl.read_csv(str(path), **kwargs)
        except Exception as e:
            raise DataForgeError(f"Failed to read CSV with Polars: {e}") from e

    def read_parquet(
        self,
        path: Union[str, Path, List[str]],
        options: Optional[ReadOptions] = None,
    ) -> "pl.DataFrame":
        options = options or ReadOptions()
        try:
            kwargs: Dict[str, Any] = {}
            if options.columns:
                kwargs["columns"] = options.columns
            if self.config.use_lazy:
                lf = pl.scan_parquet(str(path), **kwargs)
                result: pl.DataFrame = lf.collect()  # type: ignore[assignment]
                if options.sample_fraction is not None:
                    return result.sample(fraction=options.sample_fraction)
                return result
            return pl.read_parquet(str(path), **kwargs)
        except Exception as e:
            raise DataForgeError(f"Failed to read Parquet with Polars: {e}") from e

    def read_json(
        self,
        path: Union[str, Path, List[str]],
        options: Optional[ReadOptions] = None,
    ) -> "pl.DataFrame":
        try:
            return pl.read_json(str(path))
        except Exception as e:
            raise DataForgeError(f"Failed to read JSON with Polars: {e}") from e

    def read_delta(
        self,
        path: Union[str, Path],
        options: Optional[ReadOptions] = None,
        version: Optional[int] = None,
        timestamp: Optional[str] = None,
    ) -> "pl.DataFrame":
        try:
            kwargs: Dict[str, Any] = {}
            if version is not None:
                kwargs["version"] = version
            return pl.read_delta(str(path), **kwargs)
        except Exception as e:
            raise DataForgeError(f"Failed to read Delta with Polars: {e}") from e

    # ==========================================================================
    # WRITE OPERATIONS
    # ==========================================================================

    def write_csv(
        self,
        df: "pl.DataFrame",
        path: Union[str, Path],
        options: Optional[WriteOptions] = None,
    ) -> None:
        options = options or WriteOptions()
        try:
            df.write_csv(str(path), include_header=options.header)
        except Exception as e:
            raise DataForgeError(f"Failed to write CSV with Polars: {e}") from e

    def write_parquet(
        self,
        df: "pl.DataFrame",
        path: Union[str, Path],
        options: Optional[WriteOptions] = None,
    ) -> None:
        options = options or WriteOptions()
        try:
            kwargs: Dict[str, Any] = {}
            if options.compression:
                kwargs["compression"] = options.compression
            df.write_parquet(str(path), **kwargs)
        except Exception as e:
            raise DataForgeError(f"Failed to write Parquet with Polars: {e}") from e

    def write_delta(
        self,
        df: "pl.DataFrame",
        path: Union[str, Path],
        options: Optional[WriteOptions] = None,
    ) -> None:
        options = options or WriteOptions()
        try:
            delta_mode = cast(
                Literal["error", "append", "overwrite", "ignore"],
                options.mode,
            )
            df.write_delta(str(path), mode=delta_mode)
        except Exception as e:
            raise DataForgeError(f"Failed to write Delta with Polars: {e}") from e

    # ==========================================================================
    # TRANSFORMATION OPERATIONS
    # ==========================================================================

    def filter(self, df: "pl.DataFrame", condition: str) -> "pl.DataFrame":
        try:
            return df.sql(f"SELECT * FROM self WHERE {condition}")
        except Exception as e:
            raise TransformationError(f"Filter failed: {e}") from e

    def select(self, df: "pl.DataFrame", columns: List[str]) -> "pl.DataFrame":
        try:
            return df.select(columns)
        except Exception as e:
            raise TransformationError(f"Select failed: {e}") from e

    def rename(self, df: "pl.DataFrame", columns: Dict[str, str]) -> "pl.DataFrame":
        try:
            return df.rename(columns)
        except Exception as e:
            raise TransformationError(f"Rename failed: {e}") from e

    def with_column(
        self,
        df: "pl.DataFrame",
        name: str,
        expression: Union[str, Callable],
    ) -> "pl.DataFrame":
        try:
            if callable(expression) and not isinstance(expression, str):
                return df.with_columns(expression(df).alias(name))
            return df.with_columns(pl.sql_expr(str(expression)).alias(name))
        except Exception as e:
            raise TransformationError(f"with_column failed: {e}") from e

    def drop(self, df: "pl.DataFrame", columns: List[str]) -> "pl.DataFrame":
        try:
            return df.drop(columns)
        except Exception as e:
            raise TransformationError(f"Drop failed: {e}") from e

    def distinct(self, df: "pl.DataFrame") -> "pl.DataFrame":
        return df.unique()

    def sort(
        self,
        df: "pl.DataFrame",
        columns: List[str],
        ascending: Union[bool, List[bool]] = True,
    ) -> "pl.DataFrame":
        try:
            if isinstance(ascending, bool):
                descending = [not ascending] * len(columns)
            else:
                descending = [not a for a in ascending]
            return df.sort(columns, descending=descending)
        except Exception as e:
            raise TransformationError(f"Sort failed: {e}") from e

    def limit(self, df: "pl.DataFrame", n: int) -> "pl.DataFrame":
        return df.head(n)

    # ==========================================================================
    # AGGREGATION OPERATIONS
    # ==========================================================================

    def groupby(
        self,
        df: "pl.DataFrame",
        columns: List[str],
        aggregations: Dict[str, Union[str, List[str]]],
    ) -> "pl.DataFrame":
        try:
            agg_exprs = self._build_agg_exprs(aggregations)
            return df.group_by(columns).agg(agg_exprs)
        except Exception as e:
            raise TransformationError(f"GroupBy failed: {e}") from e

    def agg(
        self,
        df: "pl.DataFrame",
        aggregations: Dict[str, Union[str, List[str]]],
    ) -> "pl.DataFrame":
        try:
            agg_exprs = self._build_agg_exprs(aggregations)
            return df.select(agg_exprs)
        except Exception as e:
            raise TransformationError(f"Aggregation failed: {e}") from e

    @staticmethod
    def _build_agg_exprs(aggregations: Dict[str, Union[str, List[str]]]) -> list:
        """Build Polars aggregation expressions from dict specification."""
        exprs = []
        agg_method_map = {
            "sum": "sum",
            "avg": "mean",
            "mean": "mean",
            "min": "min",
            "max": "max",
            "count": "count",
            "first": "first",
            "last": "last",
            "std": "std",
            "var": "var",
        }
        for col, funcs in aggregations.items():
            if isinstance(funcs, str):
                funcs = [funcs]
            for func in funcs:
                method = agg_method_map.get(func, func)
                expr = getattr(pl.col(col), method)()
                exprs.append(expr.alias(f"{col}_{func}"))
        return exprs

    # ==========================================================================
    # JOIN OPERATIONS
    # ==========================================================================

    def join(
        self,
        left: "pl.DataFrame",
        right: "pl.DataFrame",
        on: Optional[Union[str, List[str]]] = None,
        left_on: Optional[Union[str, List[str]]] = None,
        right_on: Optional[Union[str, List[str]]] = None,
        how: JoinType = JoinType.INNER,
    ) -> "pl.DataFrame":
        try:
            polars_how = _JOIN_MAP.get(how, "inner")
            kwargs: Dict[str, Any] = {"how": polars_how}

            if on is not None:
                kwargs["on"] = on
            elif left_on is not None and right_on is not None:
                kwargs["left_on"] = left_on
                kwargs["right_on"] = right_on

            return left.join(right, **kwargs)
        except Exception as e:
            raise TransformationError(f"Join failed: {e}") from e

    # ==========================================================================
    # UTILITY OPERATIONS
    # ==========================================================================

    def count(self, df: "pl.DataFrame") -> int:
        return df.height

    def columns(self, df: "pl.DataFrame") -> List[str]:
        return df.columns

    def dtypes(self, df: "pl.DataFrame") -> Dict[str, str]:
        return {col: str(dtype) for col, dtype in zip(df.columns, df.dtypes)}

    def schema(self, df: "pl.DataFrame") -> Dict[str, Any]:
        return {col: str(dtype) for col, dtype in df.schema.items()}

    def to_pandas(self, df: "pl.DataFrame") -> "Any":
        return df.to_pandas()

    def from_pandas(self, df: "Any") -> "pl.DataFrame":
        return pl.from_pandas(df)

    def cache(self, df: "pl.DataFrame") -> "pl.DataFrame":
        # Polars DataFrames are already materialized in memory
        return df.clone()

    def uncache(self, df: "pl.DataFrame") -> "pl.DataFrame":
        return df

    def show(self, df: "pl.DataFrame", n: int = 20, truncate: bool = True) -> None:
        print(df.head(n))

    def collect(self, df: "pl.DataFrame") -> List[Dict[str, Any]]:
        return df.to_dicts()

    def head(self, df: "pl.DataFrame", n: int = 5) -> List[Dict[str, Any]]:
        return df.head(n).to_dicts()
