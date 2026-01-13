"""
DataForge Engine Implementations

This module provides concrete implementations of the DataFrameEngine interface
for different processing backends.

Available Engines:
    - PandasEngine: Single-node processing with pandas
    - SparkEngine: Distributed processing with PySpark
    - RapidsEngine: GPU-accelerated processing with RAPIDS/cuDF

Each engine implements the same interface but applies best practices specific
to its backend for optimal performance.

Engine Selection Guide:
    +---------------+----------------+------------------+-------------------+
    | Data Size     | No GPU/Cluster | With GPU         | With Cluster      |
    +---------------+----------------+------------------+-------------------+
    | < 100 MB      | Pandas         | Pandas           | Pandas            |
    | 100 MB - 1 GB | Pandas         | RAPIDS           | Pandas/Spark      |
    | 1 GB - 10 GB  | Spark (local)  | RAPIDS           | Spark             |
    | > 10 GB       | Spark          | Spark + RAPIDS   | Spark             |
    +---------------+----------------+------------------+-------------------+

Example:
    >>> from dataforge.engines import PandasEngine, SparkEngine
    >>>
    >>> # Use Pandas for small datasets
    >>> pandas_engine = PandasEngine()
    >>> df = pandas_engine.read_csv("small_data.csv")
    >>>
    >>> # Use Spark for large datasets
    >>> spark_engine = SparkEngine()
    >>> df = spark_engine.read_parquet("large_data/")

Best Practices:
    - Use Pandas for data < 1GB and complex transformations
    - Use Spark for distributed processing and data > 1GB
    - Use RAPIDS when GPU is available and data fits in GPU memory
    - Use the EngineRecommender for automatic selection
"""

from dataforge.engines.pandas_engine import PandasEngine
from dataforge.engines.spark_engine import SparkEngine
from dataforge.engines.rapids_engine import RapidsEngine

__all__ = [
    "PandasEngine",
    "SparkEngine",
    "RapidsEngine",
]


def get_available_engines() -> dict:
    """
    Get dictionary of available engines and their status.

    Returns:
        Dictionary mapping engine name to availability status

    Example:
        >>> engines = get_available_engines()
        >>> print(engines)
        {'pandas': True, 'spark': True, 'rapids': False}
    """
    return {
        "pandas": PandasEngine.check_availability(),
        "spark": SparkEngine.check_availability(),
        "rapids": RapidsEngine.check_availability(),
    }
