"""
DataForge Engine Comparison Example

This example demonstrates:
    - Performance comparison between engines
    - Hardware detection and recommendations
    - Benchmarking operations across engines
    - Best practices for engine selection

Run this example:
    python examples/02_engine_comparison.py

Note: Requires PySpark and optionally RAPIDS for full comparison.

Author: DataForge Team
"""

import sys
sys.path.insert(0, '.')

import time
import numpy as np
from typing import Dict, Any, Callable

from dataforge import (
    DataFrame,
    EngineRecommender,
    EngineType,
)
from dataforge.advisor.hardware_detector import HardwareDetector
from dataforge.benchmarks.profiler import Profiler
from dataforge.benchmarks.reporter import BenchmarkReporter
from dataforge.utils import setup_logging, get_logger

# Setup logging
setup_logging(level="INFO")
logger = get_logger(__name__)


def generate_sample_data(n_rows: int) -> Dict[str, Any]:
    """Generate sample data for benchmarking."""
    np.random.seed(42)

    return {
        "id": list(range(n_rows)),
        "category": np.random.choice(["A", "B", "C", "D", "E"], n_rows).tolist(),
        "value1": np.random.randn(n_rows).tolist(),
        "value2": np.random.randn(n_rows).tolist(),
        "value3": np.random.randint(1, 1000, n_rows).tolist(),
        "region": np.random.choice(
            ["North", "South", "East", "West"], n_rows
        ).tolist(),
    }


def benchmark_operation(
    name: str,
    operation: Callable,
    iterations: int = 3
) -> Dict[str, float]:
    """Benchmark an operation multiple times."""
    times = []

    for _ in range(iterations):
        start = time.perf_counter()
        result = operation()
        # Force evaluation if lazy
        if hasattr(result, 'count'):
            result.count()
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return {
        "min": min(times),
        "max": max(times),
        "avg": sum(times) / len(times),
    }


def main():
    """Main entry point for engine comparison example."""

    print("=" * 70)
    print("DataForge Engine Comparison Example")
    print("=" * 70)

    # =========================================================================
    # 1. Hardware Detection
    # =========================================================================
    print("\n1. HARDWARE DETECTION")
    print("-" * 50)

    detector = HardwareDetector()
    capabilities = detector.detect()

    print(f"  Total Memory: {capabilities.total_memory_gb:.1f} GB")
    print(f"  CPU Cores: {capabilities.cpu_cores}")
    print(f"  GPU Available: {capabilities.gpu_available}")
    if capabilities.gpu_available:
        print(f"  GPU Memory: {capabilities.gpu_memory_gb:.1f} GB")
        print(f"  GPU Name: {capabilities.gpu_name}")
    print(f"  Spark Available: {capabilities.spark_available}")

    # =========================================================================
    # 2. Engine Recommendations
    # =========================================================================
    print("\n2. ENGINE RECOMMENDATIONS")
    print("-" * 50)

    recommender = EngineRecommender()

    scenarios = [
        {"data_size_gb": 0.01, "description": "Tiny dataset (10 MB)"},
        {"data_size_gb": 0.1, "description": "Small dataset (100 MB)"},
        {"data_size_gb": 1.0, "description": "Medium dataset (1 GB)"},
        {"data_size_gb": 10.0, "description": "Large dataset (10 GB)"},
        {"data_size_gb": 100.0, "description": "Very large dataset (100 GB)"},
    ]

    print("\n  Recommendations by data size:")
    print("  " + "-" * 65)
    print(f"  {'Description':<30} {'Engine':<10} {'Reason'}")
    print("  " + "-" * 65)

    for scenario in scenarios:
        rec = recommender.recommend(data_size_gb=scenario["data_size_gb"])
        print(f"  {scenario['description']:<30} {rec.engine.value:<10} {rec.reason}")

    # =========================================================================
    # 3. Performance Comparison (Pandas vs Pandas optimized)
    # =========================================================================
    print("\n3. PERFORMANCE BENCHMARKS")
    print("-" * 50)

    # Generate test data
    data_sizes = [10_000, 100_000]

    for n_rows in data_sizes:
        print(f"\n  Dataset size: {n_rows:,} rows")
        print("  " + "-" * 40)

        data = generate_sample_data(n_rows)

        # Create DataFrame
        df = DataFrame.from_dict(data, engine="pandas")

        # Benchmark operations
        operations = {
            "Filter": lambda: df.filter("value1 > 0"),
            "Select": lambda: df.select(["id", "category", "value1"]),
            "GroupBy Sum": lambda: df.groupby("category").agg({"value1": "sum"}),
            "GroupBy Multi": lambda: df.groupby(["category", "region"]).agg({
                "value1": "mean",
                "value2": "sum",
                "value3": "count"
            }),
            "Sort": lambda: df.sort("value1", ascending=False),
            "Add Column": lambda: df.with_column("new_col", "value1 + value2"),
        }

        print(f"\n  {'Operation':<20} {'Avg (ms)':<12} {'Min (ms)':<12} {'Max (ms)'}")
        print("  " + "-" * 56)

        for op_name, operation in operations.items():
            results = benchmark_operation(op_name, operation)
            print(
                f"  {op_name:<20} "
                f"{results['avg']*1000:<12.2f} "
                f"{results['min']*1000:<12.2f} "
                f"{results['max']*1000:<12.2f}"
            )

    # =========================================================================
    # 4. Using the Built-in Profiler
    # =========================================================================
    print("\n4. BUILT-IN PROFILER")
    print("-" * 50)

    profiler = Profiler()
    data = generate_sample_data(50_000)
    df = DataFrame.from_dict(data, engine="pandas")

    # Profile operations
    with profiler.profile("filter_operation"):
        _ = df.filter("value1 > 0 AND value3 > 500")

    with profiler.profile("groupby_operation"):
        _ = df.groupby("category").agg({"value1": "mean", "value2": "sum"})

    with profiler.profile("complex_pipeline"):
        _ = (
            df
            .filter("value1 > -1")
            .select(["id", "category", "value1", "value2"])
            .with_column("ratio", "value1 / (value2 + 0.001)")
            .groupby("category")
            .agg({"ratio": "mean"})
        )

    # Print profiler results
    print("\n  Profiler Results:")
    print("  " + "-" * 40)
    results = profiler.get_results()
    for name, timing in results.items():
        print(f"  {name}: {timing['duration_ms']:.2f} ms")

    # =========================================================================
    # 5. Benchmark Reporter
    # =========================================================================
    print("\n5. BENCHMARK REPORT")
    print("-" * 50)

    reporter = BenchmarkReporter()

    # Add results
    reporter.add_result("pandas", "filter", 15.5, n_rows=100_000)
    reporter.add_result("pandas", "groupby", 45.2, n_rows=100_000)
    reporter.add_result("pandas", "join", 120.8, n_rows=100_000)
    reporter.add_result("spark", "filter", 250.0, n_rows=100_000)
    reporter.add_result("spark", "groupby", 180.5, n_rows=100_000)
    reporter.add_result("spark", "join", 95.3, n_rows=100_000)

    # Print summary
    print("\n  Benchmark Summary (simulated data):")
    summary = reporter.get_summary()
    for engine, stats in summary.items():
        print(f"\n  {engine.upper()}:")
        print(f"    Operations: {stats['total_operations']}")
        print(f"    Total time: {stats['total_time_ms']:.1f} ms")
        print(f"    Avg time: {stats['avg_time_ms']:.1f} ms")

    # =========================================================================
    # 6. Engine Selection Decision Tree
    # =========================================================================
    print("\n6. ENGINE SELECTION DECISION TREE")
    print("-" * 50)

    decision_tree = """
    Dataset Size Analysis:

    ┌─────────────────────────────────────────────────────────────────┐
    │                    DATA SIZE < 100 MB?                          │
    │                           │                                     │
    │         YES ──────────────┼────────────── NO                    │
    │          │                │                │                    │
    │          ▼                │                ▼                    │
    │    ┌──────────┐           │         DATA SIZE < 1 GB?           │
    │    │  PANDAS  │           │                │                    │
    │    │ (fastest │           │      YES ──────┼────── NO           │
    │    │  option) │           │       │        │        │           │
    │    └──────────┘           │       ▼        │        ▼           │
    │                           │  ┌──────────┐  │  ┌──────────┐      │
    │                           │  │   GPU    │  │  │  SPARK   │      │
    │                           │  │ AVAILABLE?│  │  │(required │      │
    │                           │  └──────────┘  │  │  for     │      │
    │                           │       │        │  │  scale)  │      │
    │                           │ YES ──┼── NO   │  └──────────┘      │
    │                           │  │    │    │   │                    │
    │                           │  ▼    │    ▼   │                    │
    │                           │RAPIDS │ PANDAS │                    │
    │                           │       │ /SPARK │                    │
    └─────────────────────────────────────────────────────────────────┘

    Performance Characteristics:

    │ Engine  │ Best For                      │ Avoid When              │
    ├─────────┼───────────────────────────────┼─────────────────────────┤
    │ Pandas  │ < 1GB, EDA, prototyping       │ > 1GB, streaming        │
    │ Spark   │ > 1GB, distributed, streaming │ < 100MB (overhead)      │
    │ RAPIDS  │ GPU available, ML workloads   │ No GPU, simple ops      │
    """

    print(decision_tree)

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print("""
    1. PANDAS: Best for data < 1GB on single machine
       - Low overhead, rich ecosystem
       - Use categorical dtypes for low cardinality columns
       - Enable copy-on-write for memory efficiency

    2. SPARK: Best for data > 1GB or distributed processing
       - Enable AQE for automatic optimization
       - Use broadcast joins for small tables
       - Partition appropriately (~128MB per partition)

    3. RAPIDS: Best when GPU available and data fits in GPU memory
       - 10-100x speedup for supported operations
       - Automatic CPU fallback when needed
       - Best for ML and aggregation workloads

    4. AUTO SELECTION: Let DataForge choose based on:
       - Data size
       - Available hardware
       - Operation complexity
    """)

    print("Engine comparison example completed!")


if __name__ == "__main__":
    main()
