"""
DataForge Size Analyzer Module

This module provides utilities for analyzing dataset sizes to inform
engine selection decisions.

Features:
    - File size estimation
    - Memory footprint calculation
    - Compression ratio estimation
    - Row count estimation

Supported Formats:
    - CSV (with sampling for estimation)
    - Parquet (metadata-based analysis)
    - JSON
    - Delta Lake
    - ORC

Example:
    >>> from dataforge.advisor import SizeAnalyzer
    >>>
    >>> analyzer = SizeAnalyzer()
    >>>
    >>> # Analyze file size
    >>> info = analyzer.analyze_path("data.parquet")
    >>> print(f"File size: {info.file_size_mb} MB")
    >>> print(f"Estimated rows: {info.estimated_rows}")
    >>> print(f"Memory footprint: {info.memory_footprint_mb} MB")
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union, List
import glob as glob_module


@dataclass
class SizeInfo:
    """
    Container for dataset size analysis results.

    Attributes:
        file_size_mb: Total file size in megabytes
        file_size_bytes: Total file size in bytes
        estimated_rows: Estimated number of rows
        estimated_columns: Number of columns (if detectable)
        memory_footprint_mb: Estimated memory footprint when loaded
        compression_ratio: Estimated compression ratio (for compressed formats)
        file_count: Number of files (for directories)
        format: Detected file format
    """
    file_size_mb: float
    file_size_bytes: int
    estimated_rows: Optional[int] = None
    estimated_columns: Optional[int] = None
    memory_footprint_mb: Optional[float] = None
    compression_ratio: float = 1.0
    file_count: int = 1
    format: Optional[str] = None


class SizeAnalyzer:
    """
    Dataset size analyzer for engine selection.

    Analyzes files and directories to estimate dataset sizes and
    memory requirements for different engines.

    Estimation Factors:
        - CSV: 2-5x file size in memory (dtype dependent)
        - Parquet: 3-10x file size in memory (compression ratio)
        - JSON: 3-8x file size in memory
        - Delta: Similar to Parquet + transaction log overhead

    Example:
        >>> analyzer = SizeAnalyzer()
        >>>
        >>> # Quick size check
        >>> size_mb = analyzer.get_size_mb("data/")
        >>>
        >>> # Detailed analysis
        >>> info = analyzer.analyze_path("data.parquet")
        >>> print(f"Expected memory: {info.memory_footprint_mb} MB")
    """

    # Memory multipliers by format (file size to memory)
    MEMORY_MULTIPLIERS = {
        "csv": 3.5,      # Average for CSV
        "parquet": 5.0,  # Compressed columnar
        "json": 4.0,     # JSON overhead
        "orc": 5.0,      # Similar to Parquet
        "delta": 5.0,    # Similar to Parquet
        "avro": 3.0,     # Row-based, less overhead
    }

    def analyze_path(self, path: Union[str, Path]) -> SizeInfo:
        """
        Analyze a file or directory for size information.

        Args:
            path: Path to file or directory

        Returns:
            SizeInfo with analysis results
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Path does not exist: {path}")

        if path.is_file():
            return self._analyze_file(path)
        else:
            return self._analyze_directory(path)

    def get_size_mb(self, path: Union[str, Path]) -> float:
        """
        Get total size in megabytes.

        Quick method for simple size checks.

        Args:
            path: Path to file or directory

        Returns:
            Size in megabytes
        """
        return self.analyze_path(path).file_size_mb

    def get_memory_estimate_mb(self, path: Union[str, Path]) -> float:
        """
        Estimate memory footprint when loaded.

        Args:
            path: Path to file or directory

        Returns:
            Estimated memory in megabytes
        """
        info = self.analyze_path(path)
        return info.memory_footprint_mb or info.file_size_mb * 3.5

    def _analyze_file(self, path: Path) -> SizeInfo:
        """Analyze a single file."""
        file_size_bytes = path.stat().st_size
        file_size_mb = file_size_bytes / (1024 * 1024)

        # Detect format
        file_format = self._detect_format(path)

        # Get multiplier
        multiplier = self.MEMORY_MULTIPLIERS.get(file_format, 3.5)

        # Estimate memory footprint
        memory_footprint_mb = file_size_mb * multiplier

        # Try to get row/column estimates for Parquet
        estimated_rows = None
        estimated_columns = None

        if file_format == "parquet":
            try:
                estimated_rows, estimated_columns = self._analyze_parquet_metadata(path)
            except Exception:
                pass

        return SizeInfo(
            file_size_mb=file_size_mb,
            file_size_bytes=file_size_bytes,
            estimated_rows=estimated_rows,
            estimated_columns=estimated_columns,
            memory_footprint_mb=memory_footprint_mb,
            compression_ratio=multiplier,
            file_count=1,
            format=file_format,
        )

    def _analyze_directory(self, path: Path) -> SizeInfo:
        """Analyze a directory (potentially partitioned dataset)."""
        total_size = 0
        file_count = 0
        detected_format = None

        # Check for Delta table
        if (path / "_delta_log").exists():
            detected_format = "delta"
            # For Delta, analyze only parquet files
            pattern = str(path / "**" / "*.parquet")
        else:
            # Check common file patterns
            for ext in ["parquet", "csv", "json", "orc"]:
                pattern = str(path / "**" / f"*.{ext}")
                files = glob_module.glob(pattern, recursive=True)
                if files:
                    detected_format = ext
                    break
            else:
                pattern = str(path / "**" / "*")

        # Calculate total size
        for file_path in glob_module.glob(pattern, recursive=True):
            if os.path.isfile(file_path):
                total_size += os.path.getsize(file_path)
                file_count += 1

        file_size_mb = total_size / (1024 * 1024)

        # Get multiplier
        multiplier = self.MEMORY_MULTIPLIERS.get(detected_format or "parquet", 3.5)
        memory_footprint_mb = file_size_mb * multiplier

        return SizeInfo(
            file_size_mb=file_size_mb,
            file_size_bytes=total_size,
            estimated_rows=None,
            estimated_columns=None,
            memory_footprint_mb=memory_footprint_mb,
            compression_ratio=multiplier,
            file_count=file_count,
            format=detected_format,
        )

    def _detect_format(self, path: Path) -> str:
        """Detect file format from extension."""
        suffix = path.suffix.lower().lstrip(".")

        format_mapping = {
            "parquet": "parquet",
            "pq": "parquet",
            "csv": "csv",
            "tsv": "csv",
            "json": "json",
            "jsonl": "json",
            "orc": "orc",
            "avro": "avro",
        }

        return format_mapping.get(suffix, "unknown")

    def _analyze_parquet_metadata(self, path: Path) -> tuple:
        """Extract row count and column count from Parquet metadata."""
        try:
            import pyarrow.parquet as pq

            parquet_file = pq.ParquetFile(str(path))
            metadata = parquet_file.metadata

            total_rows = metadata.num_rows
            num_columns = metadata.num_columns

            return total_rows, num_columns

        except ImportError:
            # Try pandas
            try:
                import pandas as pd
                pf = pd.read_parquet(str(path), engine="pyarrow")
                return len(pf), len(pf.columns)
            except Exception:
                pass

        return None, None

    def estimate_csv_rows(
        self,
        path: Union[str, Path],
        sample_bytes: int = 1024 * 1024
    ) -> int:
        """
        Estimate row count for CSV file by sampling.

        Args:
            path: Path to CSV file
            sample_bytes: Number of bytes to sample

        Returns:
            Estimated row count
        """
        path = Path(path)
        file_size = path.stat().st_size

        with open(path, "rb") as f:
            sample = f.read(sample_bytes)

        # Count lines in sample
        lines_in_sample = sample.count(b"\n")

        if lines_in_sample == 0:
            return 1

        # Estimate total lines
        bytes_per_line = sample_bytes / lines_in_sample
        estimated_rows = int(file_size / bytes_per_line)

        return estimated_rows
