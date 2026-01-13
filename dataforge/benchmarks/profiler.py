"""
DataForge Performance Profiler

Utilities for measuring execution time and resource usage
across different data processing engines.

Features:
    - Execution time measurement
    - Memory usage tracking
    - Multi-run averaging
    - Engine comparison

Example:
    >>> profiler = Profiler()
    >>>
    >>> with profiler.measure("operation_name"):
    ...     # Code to measure
    ...     result = expensive_operation()
    >>>
    >>> print(profiler.get_timing("operation_name"))
"""

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
import time
import logging

logger = logging.getLogger(__name__)


@dataclass
class TimingResult:
    """Result of a timing measurement.

    Attributes:
        name: Operation name
        duration_seconds: Execution time in seconds
        memory_mb: Memory usage in MB (if measured)
        iterations: Number of iterations
        timestamp: When measurement was taken
        metadata: Additional metadata
    """
    name: str
    duration_seconds: float
    memory_mb: Optional[float] = None
    iterations: int = 1
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        """Duration in milliseconds."""
        return self.duration_seconds * 1000

    @property
    def avg_duration_seconds(self) -> float:
        """Average duration per iteration."""
        return self.duration_seconds / self.iterations

    def __str__(self) -> str:
        mem_str = f", memory={self.memory_mb:.1f}MB" if self.memory_mb else ""
        return f"{self.name}: {self.duration_seconds:.3f}s{mem_str}"


class Profiler:
    """
    Performance profiler for measuring execution times.

    Example:
        >>> profiler = Profiler()
        >>>
        >>> # Context manager usage
        >>> with profiler.measure("read"):
        ...     df = pd.read_csv("data.csv")
        >>>
        >>> with profiler.measure("transform"):
        ...     result = df.groupby("col").sum()
        >>>
        >>> # Get results
        >>> print(profiler.summary())
        >>>
        >>> # Function decorator usage
        >>> @profiler.profile
        >>> def my_function():
        ...     pass
    """

    def __init__(self, track_memory: bool = False) -> None:
        """
        Initialize profiler.

        Args:
            track_memory: Enable memory usage tracking
        """
        self.track_memory = track_memory
        self._timings: Dict[str, List[TimingResult]] = {}

    @contextmanager
    def measure(
        self,
        name: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Context manager for measuring execution time.

        Args:
            name: Operation name
            metadata: Optional metadata to record

        Example:
            >>> with profiler.measure("my_operation"):
            ...     do_something()
        """
        memory_before = None
        if self.track_memory:
            memory_before = self._get_memory_usage()

        start_time = time.perf_counter()

        try:
            yield
        finally:
            duration = time.perf_counter() - start_time

            memory_used = None
            if self.track_memory and memory_before is not None:
                memory_after = self._get_memory_usage()
                memory_used = memory_after - memory_before

            result = TimingResult(
                name=name,
                duration_seconds=duration,
                memory_mb=memory_used,
                metadata=metadata or {}
            )

            if name not in self._timings:
                self._timings[name] = []
            self._timings[name].append(result)

            logger.debug(f"Measured {name}: {duration:.3f}s")

    def profile(self, func: Callable) -> Callable:
        """
        Decorator for profiling functions.

        Args:
            func: Function to profile

        Returns:
            Wrapped function

        Example:
            >>> @profiler.profile
            >>> def my_function(x):
            ...     return x * 2
        """
        func_name = getattr(func, "__name__", "unknown")

        def wrapper(*args, **kwargs):
            with self.measure(func_name):
                return func(*args, **kwargs)
        return wrapper

    def get_timing(self, name: str) -> Optional[TimingResult]:
        """
        Get the latest timing for an operation.

        Args:
            name: Operation name

        Returns:
            Latest TimingResult or None
        """
        if name in self._timings and self._timings[name]:
            return self._timings[name][-1]
        return None

    def get_all_timings(self, name: str) -> List[TimingResult]:
        """
        Get all timings for an operation.

        Args:
            name: Operation name

        Returns:
            List of TimingResult objects
        """
        return self._timings.get(name, [])

    def get_average(self, name: str) -> Optional[float]:
        """
        Get average execution time for an operation.

        Args:
            name: Operation name

        Returns:
            Average duration in seconds or None
        """
        timings = self.get_all_timings(name)
        if not timings:
            return None
        return sum(t.duration_seconds for t in timings) / len(timings)

    def get_total(self, name: str) -> Optional[float]:
        """
        Get total execution time for an operation.

        Args:
            name: Operation name

        Returns:
            Total duration in seconds or None
        """
        timings = self.get_all_timings(name)
        if not timings:
            return None
        return sum(t.duration_seconds for t in timings)

    def clear(self, name: Optional[str] = None) -> None:
        """
        Clear timing data.

        Args:
            name: Operation name to clear (None = all)
        """
        if name:
            self._timings.pop(name, None)
        else:
            self._timings.clear()

    def summary(self) -> str:
        """
        Get human-readable summary of all timings.

        Returns:
            Formatted summary string
        """
        lines = [
            "=" * 60,
            "PERFORMANCE PROFILE",
            "=" * 60,
        ]

        if not self._timings:
            lines.append("No timings recorded")
        else:
            total_time = 0
            for name, timings in sorted(self._timings.items()):
                avg = sum(t.duration_seconds for t in timings) / len(timings)
                total = sum(t.duration_seconds for t in timings)
                total_time += total

                lines.append(
                    f"  {name}: {avg:.3f}s avg "
                    f"({len(timings)} runs, {total:.3f}s total)"
                )

            lines.append("-" * 60)
            lines.append(f"  Total time: {total_time:.3f}s")

        lines.append("=" * 60)
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """
        Export timings to dictionary.

        Returns:
            Dictionary of timing data
        """
        result = {}
        for name, timings in self._timings.items():
            result[name] = {
                "count": len(timings),
                "total_seconds": sum(t.duration_seconds for t in timings),
                "avg_seconds": sum(t.duration_seconds for t in timings) / len(timings),
                "min_seconds": min(t.duration_seconds for t in timings),
                "max_seconds": max(t.duration_seconds for t in timings),
                "timings": [
                    {
                        "duration_seconds": t.duration_seconds,
                        "memory_mb": t.memory_mb,
                        "timestamp": t.timestamp,
                    }
                    for t in timings
                ],
            }
        return result

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except ImportError:
            return 0.0


def profile_function(
    func: Callable,
    *args,
    iterations: int = 1,
    warmup: int = 0,
    **kwargs
) -> TimingResult:
    """
    Profile a function with multiple iterations.

    Args:
        func: Function to profile
        *args: Function arguments
        iterations: Number of iterations
        warmup: Number of warmup iterations
        **kwargs: Function keyword arguments

    Returns:
        TimingResult with average timing

    Example:
        >>> def my_function(x, y):
        ...     return x + y
        >>>
        >>> result = profile_function(my_function, 1, 2, iterations=100)
        >>> print(f"Average: {result.avg_duration_seconds:.6f}s")
    """
    # Warmup runs
    for _ in range(warmup):
        func(*args, **kwargs)

    # Timed runs
    start_time = time.perf_counter()
    for _ in range(iterations):
        func(*args, **kwargs)
    total_time = time.perf_counter() - start_time

    func_name = getattr(func, "__name__", "unknown")
    return TimingResult(
        name=func_name,
        duration_seconds=total_time,
        iterations=iterations
    )


def compare_engines(
    operation: Callable[[Any], Any],
    data_generators: Dict[str, Callable[[], Any]],
    iterations: int = 3,
    warmup: int = 1
) -> Dict[str, TimingResult]:
    """
    Compare operation performance across different engines.

    Args:
        operation: Operation to benchmark (takes data, returns result)
        data_generators: Dict of engine name to data generator function
        iterations: Number of iterations per engine
        warmup: Number of warmup iterations

    Returns:
        Dictionary of engine name to TimingResult

    Example:
        >>> def sum_operation(df):
        ...     return df["value"].sum()
        >>>
        >>> generators = {
        ...     "pandas": lambda: pd.DataFrame({"value": range(1000000)}),
        ...     "cudf": lambda: cudf.DataFrame({"value": range(1000000)}),
        ... }
        >>>
        >>> results = compare_engines(sum_operation, generators)
        >>> for engine, timing in results.items():
        ...     print(f"{engine}: {timing.avg_duration_seconds:.3f}s")
    """
    results = {}

    for engine_name, generator in data_generators.items():
        logger.info(f"Benchmarking {engine_name}...")

        try:
            # Generate data
            data = generator()

            # Warmup
            for _ in range(warmup):
                operation(data)

            # Timed runs
            times = []
            for _ in range(iterations):
                start = time.perf_counter()
                operation(data)
                times.append(time.perf_counter() - start)

            results[engine_name] = TimingResult(
                name=engine_name,
                duration_seconds=sum(times),
                iterations=iterations,
                metadata={
                    "min": min(times),
                    "max": max(times),
                    "avg": sum(times) / len(times),
                }
            )

        except Exception as e:
            logger.warning(f"Failed to benchmark {engine_name}: {e}")
            results[engine_name] = TimingResult(
                name=engine_name,
                duration_seconds=float('inf'),
                metadata={"error": str(e)}
            )

    return results
