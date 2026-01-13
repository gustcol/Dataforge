r"""
DataForge Quality Checks

Pre-defined data quality checks for common validation scenarios.

Check Types:
    - Completeness: No null values in required columns
    - Uniqueness: No duplicate values in key columns
    - Validity: Values within expected ranges/patterns
    - Consistency: Cross-column validation rules
    - Timeliness: Data freshness checks

Example:
    >>> from dataforge.quality import QualityCheck, run_quality_checks
    >>>
    >>> checks = [
    ...     QualityCheck.not_null("user_id"),
    ...     QualityCheck.unique("email"),
    ...     QualityCheck.in_range("age", 0, 150),
    ...     QualityCheck.matches_pattern("email", r".+@.+\..+"),
    ... ]
    >>>
    >>> results = run_quality_checks(df, checks)
    >>> for result in results:
    ...     print(f"{result.check_name}: {'PASS' if result.passed else 'FAIL'}")
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union
import re


@dataclass
class CheckResult:
    """
    Result of a quality check.

    Attributes:
        check_name: Name of the check
        passed: Whether the check passed
        message: Description of result
        failed_count: Number of rows that failed
        total_count: Total rows checked
        sample_failures: Sample of failing values
    """
    check_name: str
    passed: bool
    message: str
    failed_count: int = 0
    total_count: int = 0
    sample_failures: Optional[List[Any]] = None

    @property
    def pass_rate(self) -> float:
        """Calculate pass rate percentage."""
        if self.total_count == 0:
            return 100.0
        return ((self.total_count - self.failed_count) / self.total_count) * 100


class QualityCheck:
    """
    Data quality check definition.

    Factory methods create common check types. Custom checks can be
    created by subclassing or using the custom() factory.

    Example:
        >>> check = QualityCheck.not_null("user_id")
        >>> check = QualityCheck.in_range("amount", min_val=0)
        >>> check = QualityCheck.custom(
        ...     "positive_balance",
        ...     lambda df: df["balance"] >= 0
        ... )
    """

    def __init__(
        self,
        name: str,
        column: Optional[str],
        check_func: Callable,
        description: str = ""
    ) -> None:
        """
        Initialize quality check.

        Args:
            name: Check name
            column: Column to check (None for row-level checks)
            check_func: Function that performs the check
            description: Human-readable description
        """
        self.name = name
        self.column = column
        self.check_func = check_func
        self.description = description

    def run(self, df: Any) -> CheckResult:
        """
        Run the check on a DataFrame.

        Args:
            df: DataFrame to check

        Returns:
            CheckResult with outcome
        """
        return self.check_func(df, self.column, self.name)

    # =========================================================================
    # FACTORY METHODS FOR COMMON CHECKS
    # =========================================================================

    @classmethod
    def not_null(cls, column: str) -> "QualityCheck":
        """
        Check that column has no null values.

        Args:
            column: Column name

        Returns:
            QualityCheck instance
        """
        def check_not_null(df, col, name):
            series = _get_column(df, col)
            null_count = int(series.isna().sum())
            total = len(series)
            passed = null_count == 0

            return CheckResult(
                check_name=name,
                passed=passed,
                message=f"Column '{col}' has {null_count} null values",
                failed_count=null_count,
                total_count=total
            )

        return cls(
            name=f"not_null_{column}",
            column=column,
            check_func=check_not_null,
            description=f"Column '{column}' must not contain null values"
        )

    @classmethod
    def unique(cls, column: str) -> "QualityCheck":
        """
        Check that column has unique values.

        Args:
            column: Column name

        Returns:
            QualityCheck instance
        """
        def check_unique(df, col, name):
            series = _get_column(df, col)
            dup_count = int(series.duplicated().sum())
            total = len(series)
            passed = dup_count == 0

            sample = None
            if dup_count > 0:
                dups = series[series.duplicated(keep=False)]
                sample = dups.head(5).tolist()

            return CheckResult(
                check_name=name,
                passed=passed,
                message=f"Column '{col}' has {dup_count} duplicate values",
                failed_count=dup_count,
                total_count=total,
                sample_failures=sample
            )

        return cls(
            name=f"unique_{column}",
            column=column,
            check_func=check_unique,
            description=f"Column '{column}' must have unique values"
        )

    @classmethod
    def in_range(
        cls,
        column: str,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None
    ) -> "QualityCheck":
        """
        Check that numeric values are within range.

        Args:
            column: Column name
            min_val: Minimum allowed value
            max_val: Maximum allowed value

        Returns:
            QualityCheck instance
        """
        def check_range(df, col, name):
            series = _get_column(df, col).dropna()
            total = len(series)

            failed = 0
            if min_val is not None:
                failed += int((series < min_val).sum())
            if max_val is not None:
                failed += int((series > max_val).sum())

            passed = failed == 0

            return CheckResult(
                check_name=name,
                passed=passed,
                message=f"Column '{col}' has {failed} values outside range [{min_val}, {max_val}]",
                failed_count=failed,
                total_count=total
            )

        range_str = f"[{min_val}, {max_val}]"
        return cls(
            name=f"in_range_{column}",
            column=column,
            check_func=check_range,
            description=f"Column '{column}' values must be in range {range_str}"
        )

    @classmethod
    def in_set(cls, column: str, allowed_values: List[Any]) -> "QualityCheck":
        """
        Check that values are in allowed set.

        Args:
            column: Column name
            allowed_values: List of allowed values

        Returns:
            QualityCheck instance
        """
        def check_in_set(df, col, name):
            series = _get_column(df, col).dropna()
            total = len(series)

            invalid = ~series.isin(allowed_values)
            failed = int(invalid.sum())
            passed = failed == 0

            sample = None
            if failed > 0:
                sample = series[invalid].head(5).tolist()

            return CheckResult(
                check_name=name,
                passed=passed,
                message=f"Column '{col}' has {failed} values not in allowed set",
                failed_count=failed,
                total_count=total,
                sample_failures=sample
            )

        return cls(
            name=f"in_set_{column}",
            column=column,
            check_func=check_in_set,
            description=f"Column '{column}' values must be in {allowed_values}"
        )

    @classmethod
    def matches_pattern(cls, column: str, pattern: str) -> "QualityCheck":
        """
        Check that string values match regex pattern.

        Args:
            column: Column name
            pattern: Regex pattern

        Returns:
            QualityCheck instance
        """
        def check_pattern(df, col, name):
            series = _get_column(df, col).dropna().astype(str)
            total = len(series)

            matches = series.str.match(pattern)
            failed = int((~matches).sum())
            passed = failed == 0

            return CheckResult(
                check_name=name,
                passed=passed,
                message=f"Column '{col}' has {failed} values not matching pattern",
                failed_count=failed,
                total_count=total
            )

        return cls(
            name=f"matches_pattern_{column}",
            column=column,
            check_func=check_pattern,
            description=f"Column '{column}' must match pattern '{pattern}'"
        )

    @classmethod
    def row_count_between(
        cls,
        min_rows: Optional[int] = None,
        max_rows: Optional[int] = None
    ) -> "QualityCheck":
        """
        Check that row count is within expected range.

        Args:
            min_rows: Minimum expected rows
            max_rows: Maximum expected rows

        Returns:
            QualityCheck instance
        """
        def check_row_count(df, col, name):
            count = len(df)
            passed = True

            if min_rows is not None and count < min_rows:
                passed = False
            if max_rows is not None and count > max_rows:
                passed = False

            return CheckResult(
                check_name=name,
                passed=passed,
                message=f"Row count {count} (expected: {min_rows}-{max_rows})",
                failed_count=0 if passed else 1,
                total_count=1
            )

        return cls(
            name="row_count",
            column=None,
            check_func=check_row_count,
            description=f"Row count must be between {min_rows} and {max_rows}"
        )

    @classmethod
    def custom(
        cls,
        name: str,
        condition: Callable[[Any], Any],
        column: Optional[str] = None,
        description: str = ""
    ) -> "QualityCheck":
        """
        Create custom quality check.

        Args:
            name: Check name
            condition: Function that returns boolean mask or single bool
            column: Column to check (optional)
            description: Description

        Returns:
            QualityCheck instance
        """
        def check_custom(df, col, check_name):
            try:
                result = condition(df)

                if isinstance(result, bool):
                    passed = result
                    failed = 0 if passed else 1
                    total = 1
                else:
                    # Assume it's a boolean mask
                    total = len(result)
                    failed = int((~result).sum())
                    passed = failed == 0

                return CheckResult(
                    check_name=check_name,
                    passed=passed,
                    message=f"Custom check: {failed} failures",
                    failed_count=failed,
                    total_count=total
                )
            except Exception as e:
                return CheckResult(
                    check_name=check_name,
                    passed=False,
                    message=f"Custom check failed with error: {e}",
                    failed_count=1,
                    total_count=1
                )

        return cls(
            name=name,
            column=column,
            check_func=check_custom,
            description=description or f"Custom check: {name}"
        )


def _get_column(df: Any, column: str) -> Any:
    """Get column from DataFrame (handles Spark/pandas)."""
    if hasattr(df, 'toPandas'):
        return df.select(column).toPandas()[column]
    elif hasattr(df, 'to_pandas'):
        return df[column].to_pandas()
    return df[column]


def run_quality_checks(
    df: Any,
    checks: List[QualityCheck]
) -> List[CheckResult]:
    """
    Run multiple quality checks on a DataFrame.

    Args:
        df: DataFrame to check
        checks: List of QualityCheck instances

    Returns:
        List of CheckResult instances

    Example:
        >>> checks = [
        ...     QualityCheck.not_null("id"),
        ...     QualityCheck.unique("email"),
        ... ]
        >>> results = run_quality_checks(df, checks)
        >>> all_passed = all(r.passed for r in results)
    """
    return [check.run(df) for check in checks]
