"""
DataForge Data Profiler

Comprehensive data profiling for quality assessment and understanding.

Profiling includes:
    - Basic statistics (count, nulls, unique values)
    - Numeric statistics (min, max, mean, std, percentiles)
    - String statistics (min/max length, patterns)
    - Distribution analysis
    - Correlation analysis
    - Outlier detection

Example:
    >>> from dataforge.quality import DataProfiler
    >>>
    >>> profiler = DataProfiler()
    >>> profile = profiler.profile(df)
    >>>
    >>> # Get summary
    >>> print(profile.summary())
    >>>
    >>> # Get column details
    >>> print(profile.column_profiles["amount"])
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import json


@dataclass
class ColumnProfile:
    """
    Profile for a single column.

    Attributes:
        name: Column name
        dtype: Data type
        count: Total row count
        null_count: Number of null values
        null_percentage: Percentage of nulls
        unique_count: Number of unique values
        unique_percentage: Percentage unique
        min: Minimum value (numeric/date)
        max: Maximum value (numeric/date)
        mean: Mean value (numeric)
        std: Standard deviation (numeric)
        percentiles: Percentile values (numeric)
        min_length: Minimum string length
        max_length: Maximum string length
        most_common: Most common values
        sample_values: Sample of actual values
    """
    name: str
    dtype: str
    count: int = 0
    null_count: int = 0
    null_percentage: float = 0.0
    unique_count: int = 0
    unique_percentage: float = 0.0
    min: Any = None
    max: Any = None
    mean: Optional[float] = None
    std: Optional[float] = None
    percentiles: Optional[Dict[str, float]] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    most_common: Optional[List[tuple]] = None
    sample_values: Optional[List[Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "dtype": self.dtype,
            "count": self.count,
            "null_count": self.null_count,
            "null_percentage": round(self.null_percentage, 2),
            "unique_count": self.unique_count,
            "unique_percentage": round(self.unique_percentage, 2),
            "min": self.min,
            "max": self.max,
            "mean": round(self.mean, 4) if self.mean else None,
            "std": round(self.std, 4) if self.std else None,
            "percentiles": self.percentiles,
            "min_length": self.min_length,
            "max_length": self.max_length,
            "most_common": self.most_common,
        }


@dataclass
class DataProfile:
    """
    Complete data profile for a DataFrame.

    Attributes:
        row_count: Total number of rows
        column_count: Total number of columns
        memory_usage_mb: Estimated memory usage
        column_profiles: Profile for each column
        correlations: Correlation matrix (numeric columns)
    """
    row_count: int = 0
    column_count: int = 0
    memory_usage_mb: float = 0.0
    column_profiles: Dict[str, ColumnProfile] = field(default_factory=dict)
    correlations: Optional[Dict[str, Dict[str, float]]] = None

    def summary(self) -> str:
        """Get human-readable summary."""
        lines = [
            "=" * 60,
            "DATA PROFILE SUMMARY",
            "=" * 60,
            f"Rows: {self.row_count:,}",
            f"Columns: {self.column_count}",
            f"Memory: {self.memory_usage_mb:.2f} MB",
            "",
            "COLUMN SUMMARY:",
            "-" * 60,
        ]

        for name, profile in self.column_profiles.items():
            lines.append(
                f"  {name}: {profile.dtype} | "
                f"nulls: {profile.null_percentage:.1f}% | "
                f"unique: {profile.unique_count:,}"
            )

        lines.append("=" * 60)
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "row_count": self.row_count,
            "column_count": self.column_count,
            "memory_usage_mb": self.memory_usage_mb,
            "columns": {
                name: profile.to_dict()
                for name, profile in self.column_profiles.items()
            },
            "correlations": self.correlations,
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)


class DataProfiler:
    """
    Data profiling engine.

    Generates comprehensive profiles for DataFrames including
    statistics, distributions, and quality metrics.

    Example:
        >>> profiler = DataProfiler()
        >>> profile = profiler.profile(df)
        >>> print(profile.summary())
    """

    def __init__(
        self,
        sample_size: Optional[int] = None,
        include_correlations: bool = True,
        top_n_values: int = 10
    ) -> None:
        """
        Initialize profiler.

        Args:
            sample_size: Sample size for large datasets (None = all)
            include_correlations: Calculate correlation matrix
            top_n_values: Number of top values to track
        """
        self.sample_size = sample_size
        self.include_correlations = include_correlations
        self.top_n_values = top_n_values

    def profile(self, df: Any) -> DataProfile:
        """
        Generate profile for DataFrame.

        Args:
            df: DataFrame to profile (pandas, Spark, or cuDF)

        Returns:
            DataProfile with comprehensive statistics
        """
        # Convert to pandas for profiling
        if hasattr(df, 'toPandas'):
            pandas_df = df.toPandas()
        elif hasattr(df, 'to_pandas'):
            pandas_df = df.to_pandas()
        else:
            pandas_df = df

        # Sample if needed
        if self.sample_size and len(pandas_df) > self.sample_size:
            pandas_df = pandas_df.sample(n=self.sample_size, random_state=42)

        # Basic info
        profile = DataProfile(
            row_count=len(pandas_df),
            column_count=len(pandas_df.columns),
            memory_usage_mb=pandas_df.memory_usage(deep=True).sum() / (1024 * 1024)
        )

        # Profile each column
        for col in pandas_df.columns:
            profile.column_profiles[col] = self._profile_column(
                pandas_df[col], col
            )

        # Calculate correlations for numeric columns
        if self.include_correlations:
            numeric_cols = pandas_df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 1:
                corr_matrix = pandas_df[numeric_cols].corr()
                profile.correlations = corr_matrix.to_dict()

        return profile

    def _profile_column(self, series: Any, name: str) -> ColumnProfile:
        """Profile a single column."""
        import pandas as pd
        import numpy as np

        profile = ColumnProfile(
            name=name,
            dtype=str(series.dtype),
            count=len(series),
            null_count=int(series.isna().sum()),
            unique_count=int(series.nunique())
        )

        # Calculate percentages
        if profile.count > 0:
            profile.null_percentage = (profile.null_count / profile.count) * 100
            profile.unique_percentage = (profile.unique_count / profile.count) * 100

        # Get non-null data
        non_null = series.dropna()

        if len(non_null) == 0:
            return profile

        # Type-specific profiling
        if pd.api.types.is_numeric_dtype(series):
            profile = self._profile_numeric(profile, non_null)
        elif pd.api.types.is_datetime64_any_dtype(series):
            profile = self._profile_datetime(profile, non_null)
        else:
            profile = self._profile_string(profile, non_null)

        # Most common values
        try:
            value_counts = non_null.value_counts().head(self.top_n_values)
            profile.most_common = list(zip(
                value_counts.index.tolist(),
                value_counts.values.tolist()
            ))
        except Exception:
            pass

        # Sample values
        try:
            profile.sample_values = non_null.head(5).tolist()
        except Exception:
            pass

        return profile

    def _profile_numeric(
        self,
        profile: ColumnProfile,
        series: Any
    ) -> ColumnProfile:
        """Profile numeric column."""
        import numpy as np

        profile.min = float(series.min())
        profile.max = float(series.max())
        profile.mean = float(series.mean())
        profile.std = float(series.std())

        # Percentiles
        try:
            percentiles = series.quantile([0.25, 0.5, 0.75])
            profile.percentiles = {
                "25%": float(percentiles.iloc[0]),
                "50%": float(percentiles.iloc[1]),
                "75%": float(percentiles.iloc[2]),
            }
        except Exception:
            pass

        return profile

    def _profile_datetime(
        self,
        profile: ColumnProfile,
        series: Any
    ) -> ColumnProfile:
        """Profile datetime column."""
        profile.min = series.min()
        profile.max = series.max()
        return profile

    def _profile_string(
        self,
        profile: ColumnProfile,
        series: Any
    ) -> ColumnProfile:
        """Profile string column."""
        lengths = series.astype(str).str.len()
        profile.min_length = int(lengths.min())
        profile.max_length = int(lengths.max())
        return profile

    def compare_profiles(
        self,
        profile1: DataProfile,
        profile2: DataProfile
    ) -> Dict[str, Any]:
        """
        Compare two data profiles.

        Useful for detecting data drift between datasets.

        Args:
            profile1: First profile
            profile2: Second profile

        Returns:
            Dictionary with comparison results
        """
        comparison = {
            "row_count_diff": profile2.row_count - profile1.row_count,
            "column_changes": {
                "added": set(profile2.column_profiles.keys()) - set(profile1.column_profiles.keys()),
                "removed": set(profile1.column_profiles.keys()) - set(profile2.column_profiles.keys()),
            },
            "column_diffs": {}
        }

        # Compare common columns
        common_cols = set(profile1.column_profiles.keys()) & set(profile2.column_profiles.keys())

        for col in common_cols:
            p1 = profile1.column_profiles[col]
            p2 = profile2.column_profiles[col]

            diff = {}

            if p1.null_percentage != p2.null_percentage:
                diff["null_percentage_change"] = p2.null_percentage - p1.null_percentage

            if p1.unique_percentage != p2.unique_percentage:
                diff["unique_percentage_change"] = p2.unique_percentage - p1.unique_percentage

            if p1.mean and p2.mean:
                diff["mean_change"] = p2.mean - p1.mean

            if diff:
                comparison["column_diffs"][col] = diff

        return comparison
