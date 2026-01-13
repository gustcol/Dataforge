"""
DataForge Basic Usage Example

This example demonstrates the fundamental features of DataForge:
    - Automatic engine selection
    - Unified DataFrame API
    - Engine-specific escape hatches
    - Basic transformations

Run this example:
    python examples/01_basic_usage.py

Author: DataForge Team
"""

import sys
sys.path.insert(0, '.')

from dataforge import (
    DataFrame,
    EngineRecommender,
    EngineRecommendation,
    DataForgeConfig,
    EngineType,
)
from dataforge.utils import setup_logging, get_logger

# Setup logging
setup_logging(level="INFO")
logger = get_logger(__name__)


def main():
    """Main entry point for basic usage example."""

    print("=" * 60)
    print("DataForge Basic Usage Example")
    print("=" * 60)

    # =========================================================================
    # 1. Engine Recommendation
    # =========================================================================
    print("\n1. ENGINE RECOMMENDATION")
    print("-" * 40)

    recommender = EngineRecommender()

    # Get recommendations for different data sizes
    sizes = [0.05, 0.5, 2.0, 15.0, 150.0]  # GB

    for size in sizes:
        recommendation = recommender.recommend(data_size_gb=size)
        print(f"  {size:6.1f} GB -> {recommendation.engine.value:8s} | {recommendation.reason}")

    # =========================================================================
    # 2. Creating DataFrames with Different Engines
    # =========================================================================
    print("\n2. CREATING DATAFRAMES")
    print("-" * 40)

    # Sample data
    sample_data = {
        "id": [1, 2, 3, 4, 5],
        "name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
        "age": [30, 25, 35, 28, 32],
        "salary": [75000, 65000, 85000, 70000, 80000],
        "department": ["Engineering", "Marketing", "Engineering", "Sales", "Marketing"],
    }

    # Create DataFrame with auto engine selection (defaults to Pandas for small data)
    print("  Creating DataFrame with auto engine selection...")
    df = DataFrame.from_dict(sample_data, engine="auto")
    print(f"  Engine used: {df.engine_type.value}")
    print(f"  Row count: {df.count()}")

    # Explicitly use Pandas engine
    print("\n  Creating DataFrame with explicit Pandas engine...")
    df_pandas = DataFrame.from_dict(sample_data, engine="pandas")
    print(f"  Engine used: {df_pandas.engine_type.value}")

    # =========================================================================
    # 3. Basic Transformations
    # =========================================================================
    print("\n3. BASIC TRANSFORMATIONS")
    print("-" * 40)

    # Filter operation
    print("  Filtering employees with age > 28...")
    df_filtered = df.filter("age > 28")
    print(f"  Rows after filter: {df_filtered.count()}")

    # Select columns
    print("\n  Selecting specific columns...")
    df_selected = df.select(["name", "department", "salary"])
    print(f"  Columns: {df_selected.columns}")

    # Add new column
    print("\n  Adding calculated column...")
    df_with_bonus = df.with_column("bonus", "salary * 0.1")
    print(f"  New columns: {df_with_bonus.columns}")

    # =========================================================================
    # 4. Aggregations
    # =========================================================================
    print("\n4. AGGREGATIONS")
    print("-" * 40)

    # Group by and aggregate
    print("  Grouping by department and calculating stats...")
    df_agg = df.groupby("department").agg({
        "salary": "mean",
        "age": "max",
        "id": "count"
    })

    # Show results
    print("\n  Aggregation results:")
    native_df = df_agg.to_pandas()
    print(native_df.to_string(index=False))

    # =========================================================================
    # 5. Method Chaining
    # =========================================================================
    print("\n5. METHOD CHAINING")
    print("-" * 40)

    print("  Building complex transformation pipeline...")
    result = (
        df
        .filter("age >= 28")
        .select(["name", "department", "salary"])
        .with_column("salary_category", "CASE WHEN salary > 75000 THEN 'High' ELSE 'Standard' END")
        .sort("salary", ascending=False)
    )

    print("\n  Pipeline result:")
    print(result.to_pandas().to_string(index=False))

    # =========================================================================
    # 6. Native Engine Access (Escape Hatches)
    # =========================================================================
    print("\n6. NATIVE ENGINE ACCESS")
    print("-" * 40)

    print("  Accessing native Pandas DataFrame...")
    native_pandas = df.to_pandas()
    print(f"  Type: {type(native_pandas).__name__}")

    # Use native Pandas functionality
    print("\n  Using native Pandas operations:")
    print(f"  DataFrame shape: {native_pandas.shape}")
    print(f"  Memory usage: {native_pandas.memory_usage(deep=True).sum() / 1024:.2f} KB")
    print(f"  Salary std: {native_pandas['salary'].std():.2f}")

    # =========================================================================
    # 7. Schema Information
    # =========================================================================
    print("\n7. SCHEMA INFORMATION")
    print("-" * 40)

    print("  DataFrame schema:")
    schema = df.schema
    for col_name, col_type in schema.items():
        print(f"    {col_name}: {col_type}")

    # =========================================================================
    # 8. Configuration
    # =========================================================================
    print("\n8. CONFIGURATION")
    print("-" * 40)

    config = DataForgeConfig()
    print(f"  Default engine: {config.default_engine}")
    print(f"  Auto select: {config.auto_select_engine}")
    print(f"  Pandas max GB: {config.pandas_config.max_memory_gb}")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("""
DataForge provides:
  - Unified API for Pandas, Spark, and RAPIDS
  - Automatic engine selection based on data size
  - Method chaining for readable transformations
  - Native engine access when needed
  - Consistent behavior across engines
    """)

    print("Example completed successfully!")


if __name__ == "__main__":
    main()
