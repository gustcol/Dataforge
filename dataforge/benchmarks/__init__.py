"""
DataForge Benchmarks Module

Performance benchmarking utilities for comparing engines
and measuring execution times.

Features:
    - Execution time profiling
    - Memory usage tracking
    - Engine comparison benchmarks
    - Report generation

Example:
    >>> from dataforge.benchmarks import Profiler, BenchmarkReporter
    >>>
    >>> # Profile operations
    >>> profiler = Profiler()
    >>> with profiler.measure("read_csv"):
    ...     df = pd.read_csv("data.csv")
    >>>
    >>> with profiler.measure("transform"):
    ...     df = df.groupby("category").sum()
    >>>
    >>> # Generate report
    >>> report = profiler.report()
    >>> print(report.summary())
"""

from dataforge.benchmarks.profiler import (
    Profiler,
    TimingResult,
    profile_function,
    compare_engines,
)
from dataforge.benchmarks.reporter import (
    BenchmarkReporter,
    BenchmarkResult,
    generate_comparison_report,
)

__all__ = [
    # Profiler
    "Profiler",
    "TimingResult",
    "profile_function",
    "compare_engines",
    # Reporter
    "BenchmarkReporter",
    "BenchmarkResult",
    "generate_comparison_report",
]
