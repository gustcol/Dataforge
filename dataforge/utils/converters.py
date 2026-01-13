"""
DataForge Data Converters

Utilities for converting between different DataFrame types
and data formats.

Features:
    - DataFrame type conversion (pandas, Spark, cuDF)
    - Schema conversion
    - Engine type inference
    - Data format conversion

Example:
    >>> from dataforge.utils import convert_to_pandas, convert_to_spark
    >>>
    >>> # Convert Spark DataFrame to pandas
    >>> pandas_df = convert_to_pandas(spark_df)
    >>>
    >>> # Convert pandas to Spark
    >>> spark_df = convert_to_spark(pandas_df, spark_session)
"""

from typing import Any, Dict, List, Optional, TYPE_CHECKING
import logging

if TYPE_CHECKING:
    from pyspark.sql import SparkSession, DataFrame as SparkDataFrame
    from pyspark.sql.types import StructType
    import pandas as pd

logger = logging.getLogger(__name__)


def infer_engine_type(df: Any) -> str:
    """
    Infer the engine type from a DataFrame.

    Args:
        df: DataFrame object

    Returns:
        Engine type string: "pandas", "spark", "cudf", or "unknown"

    Example:
        >>> engine = infer_engine_type(my_df)
        >>> print(f"DataFrame is from: {engine}")
    """
    df_type = type(df).__module__

    if "pandas" in df_type:
        return "pandas"
    elif "pyspark" in df_type:
        return "spark"
    elif "cudf" in df_type:
        return "cudf"
    else:
        return "unknown"


def convert_to_pandas(df: Any) -> "pd.DataFrame":
    """
    Convert any DataFrame to pandas DataFrame.

    Args:
        df: Source DataFrame (Spark, cuDF, or pandas)

    Returns:
        pandas DataFrame

    Example:
        >>> pandas_df = convert_to_pandas(spark_df)
    """
    import pandas as pd

    engine = infer_engine_type(df)

    if engine == "pandas":
        return df

    elif engine == "spark":
        logger.debug("Converting Spark DataFrame to pandas")
        return df.toPandas()

    elif engine == "cudf":
        logger.debug("Converting cuDF DataFrame to pandas")
        return df.to_pandas()

    else:
        raise TypeError(f"Cannot convert {type(df)} to pandas DataFrame")


def convert_to_spark(
    df: Any,
    spark: "SparkSession",
    schema: Optional["StructType"] = None
) -> "SparkDataFrame":
    """
    Convert any DataFrame to Spark DataFrame.

    Args:
        df: Source DataFrame (pandas, cuDF, or Spark)
        spark: SparkSession
        schema: Optional schema for conversion

    Returns:
        Spark DataFrame

    Example:
        >>> spark_df = convert_to_spark(pandas_df, spark)
    """
    engine = infer_engine_type(df)

    if engine == "spark":
        return df

    elif engine == "pandas":
        logger.debug("Converting pandas DataFrame to Spark")
        if schema:
            return spark.createDataFrame(df, schema)
        return spark.createDataFrame(df)

    elif engine == "cudf":
        logger.debug("Converting cuDF DataFrame to Spark via pandas")
        pandas_df = df.to_pandas()
        if schema:
            return spark.createDataFrame(pandas_df, schema)
        return spark.createDataFrame(pandas_df)

    else:
        raise TypeError(f"Cannot convert {type(df)} to Spark DataFrame")


def convert_to_cudf(df: Any) -> Any:
    """
    Convert any DataFrame to cuDF DataFrame.

    Requires RAPIDS cuDF to be installed.

    Args:
        df: Source DataFrame (pandas, Spark, or cuDF)

    Returns:
        cuDF DataFrame

    Example:
        >>> cudf_df = convert_to_cudf(pandas_df)
    """
    try:
        import cudf
    except ImportError:
        raise ImportError(
            "cuDF not installed. Install RAPIDS with: "
            "conda install -c rapidsai -c conda-forge cudf"
        )

    engine = infer_engine_type(df)

    if engine == "cudf":
        return df

    elif engine == "pandas":
        logger.debug("Converting pandas DataFrame to cuDF")
        return cudf.DataFrame.from_pandas(df)

    elif engine == "spark":
        logger.debug("Converting Spark DataFrame to cuDF via pandas")
        pandas_df = df.toPandas()
        return cudf.DataFrame.from_pandas(pandas_df)

    else:
        raise TypeError(f"Cannot convert {type(df)} to cuDF DataFrame")


def convert_schema(
    schema: Any,
    target: str = "spark"
) -> Any:
    """
    Convert schema between different formats.

    Args:
        schema: Source schema (dict, Spark StructType, pandas dtypes)
        target: Target format ("spark", "pandas", "dict")

    Returns:
        Converted schema

    Example:
        >>> # Dict to Spark schema
        >>> spark_schema = convert_schema(
        ...     {"id": "int", "name": "string"},
        ...     target="spark"
        ... )
    """
    if target == "spark":
        return _to_spark_schema(schema)
    elif target == "pandas":
        return _to_pandas_dtypes(schema)
    elif target == "dict":
        return _to_dict_schema(schema)
    else:
        raise ValueError(f"Unknown target format: {target}")


def _to_spark_schema(schema: Any) -> "StructType":
    """Convert schema to Spark StructType."""
    from pyspark.sql.types import (
        StructType, StructField,
        StringType, IntegerType, LongType, FloatType, DoubleType,
        BooleanType, TimestampType, DateType, BinaryType
    )

    type_mapping = {
        "string": StringType(),
        "str": StringType(),
        "int": IntegerType(),
        "integer": IntegerType(),
        "long": LongType(),
        "bigint": LongType(),
        "float": FloatType(),
        "double": DoubleType(),
        "boolean": BooleanType(),
        "bool": BooleanType(),
        "timestamp": TimestampType(),
        "date": DateType(),
        "binary": BinaryType(),
    }

    if isinstance(schema, StructType):
        return schema

    if isinstance(schema, dict):
        fields = []
        for name, dtype in schema.items():
            dtype_str = str(dtype).lower()
            spark_type = type_mapping.get(dtype_str, StringType())
            fields.append(StructField(name, spark_type, True))
        return StructType(fields)

    raise TypeError(f"Cannot convert {type(schema)} to Spark schema")


def _to_pandas_dtypes(schema: Any) -> Dict[str, str]:
    """Convert schema to pandas dtypes dictionary."""
    from pyspark.sql.types import (
        StructType, StringType, IntegerType, LongType,
        FloatType, DoubleType, BooleanType, TimestampType, DateType
    )

    type_mapping = {
        StringType: "object",
        IntegerType: "int32",
        LongType: "int64",
        FloatType: "float32",
        DoubleType: "float64",
        BooleanType: "bool",
        TimestampType: "datetime64[ns]",
        DateType: "datetime64[ns]",
    }

    if isinstance(schema, dict):
        # Assume already in pandas format
        return schema

    if isinstance(schema, StructType):
        return {
            field.name: type_mapping.get(type(field.dataType), "object")
            for field in schema.fields
        }

    raise TypeError(f"Cannot convert {type(schema)} to pandas dtypes")


def _to_dict_schema(schema: Any) -> Dict[str, str]:
    """Convert schema to simple dictionary format."""
    from pyspark.sql.types import StructType

    if isinstance(schema, dict):
        return schema

    if isinstance(schema, StructType):
        return {
            field.name: str(field.dataType).replace("Type()", "").lower()
            for field in schema.fields
        }

    raise TypeError(f"Cannot convert {type(schema)} to dict schema")


def estimate_size_mb(df: Any) -> float:
    """
    Estimate the memory size of a DataFrame in MB.

    Args:
        df: DataFrame to estimate

    Returns:
        Estimated size in MB

    Example:
        >>> size = estimate_size_mb(my_df)
        >>> print(f"DataFrame size: {size:.2f} MB")
    """
    engine = infer_engine_type(df)

    if engine == "pandas":
        return df.memory_usage(deep=True).sum() / (1024 * 1024)

    elif engine == "spark":
        # Rough estimate based on row count and schema
        try:
            # Sample-based estimation
            sample_size = min(1000, df.count())
            if sample_size == 0:
                return 0

            sample = df.limit(sample_size).toPandas()
            sample_mb = sample.memory_usage(deep=True).sum() / (1024 * 1024)

            total_rows = df.count()
            estimated_mb = (sample_mb / sample_size) * total_rows
            return estimated_mb
        except Exception:
            # Fallback: estimate based on schema
            num_cols = len(df.columns)
            num_rows = df.count()
            # Rough estimate: 100 bytes per cell average
            return (num_cols * num_rows * 100) / (1024 * 1024)

    elif engine == "cudf":
        # cuDF provides direct memory info
        try:
            return df.memory_usage(deep=True).sum() / (1024 * 1024)
        except Exception:
            # Fallback
            return df.to_pandas().memory_usage(deep=True).sum() / (1024 * 1024)

    else:
        raise TypeError(f"Cannot estimate size for {type(df)}")


def sample_dataframe(df: Any, n: int = 1000, seed: int = 42) -> Any:
    """
    Sample rows from a DataFrame.

    Args:
        df: Source DataFrame
        n: Number of rows to sample
        seed: Random seed

    Returns:
        Sampled DataFrame of same type

    Example:
        >>> sample = sample_dataframe(large_df, n=100)
    """
    engine = infer_engine_type(df)

    if engine == "pandas":
        return df.sample(n=min(n, len(df)), random_state=seed)

    elif engine == "spark":
        total_rows = df.count()
        if total_rows == 0:
            return df
        fraction = min(1.0, n / total_rows)
        return df.sample(fraction=fraction, seed=seed)

    elif engine == "cudf":
        return df.sample(n=min(n, len(df)), random_state=seed)

    else:
        raise TypeError(f"Cannot sample {type(df)}")
