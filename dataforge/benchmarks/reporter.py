"""
DataForge Benchmark Reporter

Utilities for generating benchmark reports and visualizations.

Features:
    - Comparison tables
    - Performance summaries
    - Recommendations based on results
    - Export to various formats

Example:
    >>> reporter = BenchmarkReporter()
    >>> reporter.add_result("pandas", read_time=1.5, transform_time=2.3)
    >>> reporter.add_result("spark", read_time=0.8, transform_time=1.2)
    >>> reporter.add_result("rapids", read_time=0.3, transform_time=0.5)
    >>>
    >>> print(reporter.comparison_table())
    >>> print(reporter.recommendations())
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Result from a benchmark run.

    Attributes:
        engine: Engine name
        operation: Operation name
        duration_seconds: Execution time
        data_size_mb: Data size in MB
        rows: Number of rows processed
        metadata: Additional metadata
    """
    engine: str
    operation: str
    duration_seconds: float
    data_size_mb: Optional[float] = None
    rows: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def throughput_mb_per_sec(self) -> Optional[float]:
        """Calculate throughput in MB/s."""
        if self.data_size_mb and self.duration_seconds > 0:
            return self.data_size_mb / self.duration_seconds
        return None

    @property
    def rows_per_sec(self) -> Optional[float]:
        """Calculate rows processed per second."""
        if self.rows and self.duration_seconds > 0:
            return self.rows / self.duration_seconds
        return None


class BenchmarkReporter:
    """
    Benchmark results reporter.

    Collects benchmark results and generates reports.

    Example:
        >>> reporter = BenchmarkReporter("My Benchmark Suite")
        >>>
        >>> # Add results
        >>> reporter.add_result("pandas", "read", 1.5, data_size_mb=100)
        >>> reporter.add_result("spark", "read", 0.8, data_size_mb=100)
        >>>
        >>> # Generate reports
        >>> print(reporter.summary())
        >>> print(reporter.comparison_table())
        >>>
        >>> # Export
        >>> reporter.to_json("benchmark_results.json")
    """

    def __init__(self, name: str = "Benchmark") -> None:
        """
        Initialize reporter.

        Args:
            name: Benchmark suite name
        """
        self.name = name
        self._results: List[BenchmarkResult] = []

    def add_result(
        self,
        engine: str,
        operation: str,
        duration_seconds: float,
        data_size_mb: Optional[float] = None,
        rows: Optional[int] = None,
        **metadata
    ) -> None:
        """
        Add a benchmark result.

        Args:
            engine: Engine name (pandas, spark, rapids)
            operation: Operation name
            duration_seconds: Execution time
            data_size_mb: Optional data size
            rows: Optional row count
            **metadata: Additional metadata
        """
        result = BenchmarkResult(
            engine=engine,
            operation=operation,
            duration_seconds=duration_seconds,
            data_size_mb=data_size_mb,
            rows=rows,
            metadata=metadata
        )
        self._results.append(result)

    def add_results(self, results: Dict[str, Dict[str, float]]) -> None:
        """
        Add multiple results at once.

        Args:
            results: Dict of engine -> {operation: duration}

        Example:
            >>> reporter.add_results({
            ...     "pandas": {"read": 1.5, "transform": 2.3},
            ...     "spark": {"read": 0.8, "transform": 1.2},
            ... })
        """
        for engine, operations in results.items():
            for operation, duration in operations.items():
                self.add_result(engine, operation, duration)

    def get_results(
        self,
        engine: Optional[str] = None,
        operation: Optional[str] = None
    ) -> List[BenchmarkResult]:
        """
        Get filtered results.

        Args:
            engine: Filter by engine
            operation: Filter by operation

        Returns:
            Filtered list of results
        """
        results = self._results

        if engine:
            results = [r for r in results if r.engine == engine]

        if operation:
            results = [r for r in results if r.operation == operation]

        return results

    def summary(self) -> str:
        """
        Get summary of benchmark results.

        Returns:
            Formatted summary string
        """
        lines = [
            "=" * 70,
            f"BENCHMARK REPORT: {self.name}",
            "=" * 70,
        ]

        if not self._results:
            lines.append("No results recorded")
            return "\n".join(lines)

        # Group by engine
        engines = {}
        for result in self._results:
            if result.engine not in engines:
                engines[result.engine] = []
            engines[result.engine].append(result)

        for engine, results in sorted(engines.items()):
            lines.append(f"\n{engine.upper()}")
            lines.append("-" * 40)

            total_time = 0
            for result in results:
                total_time += result.duration_seconds
                throughput = ""
                if result.throughput_mb_per_sec:
                    throughput = f" ({result.throughput_mb_per_sec:.1f} MB/s)"

                lines.append(
                    f"  {result.operation}: {result.duration_seconds:.3f}s{throughput}"
                )

            lines.append(f"  Total: {total_time:.3f}s")

        lines.append("\n" + "=" * 70)
        return "\n".join(lines)

    def comparison_table(self, operation: Optional[str] = None) -> str:
        """
        Generate comparison table.

        Args:
            operation: Filter to specific operation

        Returns:
            Formatted comparison table
        """
        results = self._results
        if operation:
            results = [r for r in results if r.operation == operation]

        if not results:
            return "No results to compare"

        # Get unique engines and operations
        engines = sorted(set(r.engine for r in results))
        operations = sorted(set(r.operation for r in results))

        # Build table
        lines = []

        # Header
        header = "Operation".ljust(20) + " | ".join(e.center(15) for e in engines)
        lines.append(header)
        lines.append("-" * len(header))

        # Data rows
        for op in operations:
            row_data = [op.ljust(20)]

            for engine in engines:
                matching = [
                    r for r in results
                    if r.engine == engine and r.operation == op
                ]

                if matching:
                    duration = matching[0].duration_seconds
                    row_data.append(f"{duration:.3f}s".center(15))
                else:
                    row_data.append("N/A".center(15))

            lines.append(" | ".join(row_data))

        # Find best performer
        lines.append("-" * len(header))

        # Calculate speedups
        if len(engines) > 1:
            base_engine = engines[0]  # Use first as baseline

            for op in operations:
                base_result = next(
                    (r for r in results if r.engine == base_engine and r.operation == op),
                    None
                )

                if base_result:
                    speedups = []
                    for engine in engines[1:]:
                        other = next(
                            (r for r in results if r.engine == engine and r.operation == op),
                            None
                        )
                        if other:
                            speedup = base_result.duration_seconds / other.duration_seconds
                            speedups.append(f"{engine}: {speedup:.1f}x vs {base_engine}")

                    if speedups:
                        lines.append(f"  {op}: " + ", ".join(speedups))

        return "\n".join(lines)

    def recommendations(self) -> str:
        """
        Generate recommendations based on results.

        Returns:
            Recommendations string
        """
        lines = [
            "=" * 60,
            "RECOMMENDATIONS",
            "=" * 60,
        ]

        if not self._results:
            lines.append("No results to analyze")
            return "\n".join(lines)

        # Find fastest engine for each operation
        operations = set(r.operation for r in self._results)

        for op in operations:
            op_results = [r for r in self._results if r.operation == op]
            if op_results:
                fastest = min(op_results, key=lambda r: r.duration_seconds)
                slowest = max(op_results, key=lambda r: r.duration_seconds)

                if len(op_results) > 1:
                    speedup = slowest.duration_seconds / fastest.duration_seconds
                    lines.append(
                        f"\n{op}:"
                        f"\n  Best: {fastest.engine} ({fastest.duration_seconds:.3f}s)"
                        f"\n  Speedup: {speedup:.1f}x vs {slowest.engine}"
                    )

        # General recommendations
        lines.append("\n" + "-" * 60)
        lines.append("General Recommendations:")

        # Check data sizes
        has_large_data = any(
            r.data_size_mb and r.data_size_mb > 1000
            for r in self._results
        )

        if has_large_data:
            lines.append("  - For datasets > 1GB, consider using Spark or RAPIDS")

        # Check if RAPIDS is fastest
        rapids_results = [r for r in self._results if r.engine.lower() == "rapids"]
        if rapids_results:
            avg_speedup = sum(
                next(
                    (p.duration_seconds for p in self._results
                     if p.engine.lower() == "pandas" and p.operation == r.operation),
                    r.duration_seconds
                ) / r.duration_seconds
                for r in rapids_results
            ) / len(rapids_results)

            if avg_speedup > 2:
                lines.append(
                    f"  - RAPIDS shows {avg_speedup:.1f}x average speedup - "
                    "consider GPU acceleration for this workload"
                )

        lines.append("=" * 60)
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """
        Export results to dictionary.

        Returns:
            Dictionary of benchmark results
        """
        return {
            "name": self.name,
            "results": [
                {
                    "engine": r.engine,
                    "operation": r.operation,
                    "duration_seconds": r.duration_seconds,
                    "data_size_mb": r.data_size_mb,
                    "rows": r.rows,
                    "throughput_mb_per_sec": r.throughput_mb_per_sec,
                    "rows_per_sec": r.rows_per_sec,
                    "metadata": r.metadata,
                }
                for r in self._results
            ],
        }

    def to_json(self, path: str, indent: int = 2) -> None:
        """
        Export results to JSON file.

        Args:
            path: Output file path
            indent: JSON indentation
        """
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=indent)
        logger.info(f"Exported benchmark results to {path}")

    def to_csv(self, path: str) -> None:
        """
        Export results to CSV file.

        Args:
            path: Output file path
        """
        import csv

        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "engine", "operation", "duration_seconds",
                "data_size_mb", "rows", "throughput_mb_per_sec"
            ])

            for r in self._results:
                writer.writerow([
                    r.engine, r.operation, r.duration_seconds,
                    r.data_size_mb, r.rows, r.throughput_mb_per_sec
                ])

        logger.info(f"Exported benchmark results to {path}")


def generate_comparison_report(
    results: Dict[str, Dict[str, float]],
    name: str = "Engine Comparison"
) -> str:
    """
    Generate a quick comparison report.

    Args:
        results: Dict of engine -> {operation: duration}
        name: Report name

    Returns:
        Formatted report string

    Example:
        >>> results = {
        ...     "pandas": {"read": 1.5, "agg": 2.0},
        ...     "spark": {"read": 0.8, "agg": 1.0},
        ... }
        >>> print(generate_comparison_report(results))
    """
    reporter = BenchmarkReporter(name)
    reporter.add_results(results)

    return "\n\n".join([
        reporter.summary(),
        reporter.comparison_table(),
        reporter.recommendations(),
    ])
