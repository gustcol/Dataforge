"""
DataForge Data Validators

Schema and column validation utilities for data quality assurance.

Validation Types:
    - Type validation: Ensure columns have correct data types
    - Null validation: Check for required/optional fields
    - Range validation: Verify numeric values are within bounds
    - Pattern validation: Match strings against regex patterns
    - Uniqueness validation: Ensure unique values
    - Referential integrity: Validate foreign key relationships

Example:
    >>> from dataforge.quality import SchemaValidator
    >>>
    >>> schema = {
    ...     "user_id": {"type": "int", "nullable": False, "unique": True},
    ...     "email": {"type": "string", "nullable": False, "pattern": r".+@.+"},
    ...     "age": {"type": "int", "min": 0, "max": 150},
    ...     "status": {"type": "string", "allowed_values": ["active", "inactive"]}
    ... }
    >>>
    >>> validator = SchemaValidator(schema)
    >>> result = validator.validate(df)
    >>> if not result.is_valid:
    ...     for error in result.errors:
    ...         print(f"Error: {error}")
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Pattern, Union


@dataclass
class ValidationResult:
    """
    Result of validation operation.

    Attributes:
        is_valid: Whether validation passed
        errors: List of error messages
        warnings: List of warning messages
        stats: Additional statistics
    """
    is_valid: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)

    def add_error(self, message: str) -> None:
        """Add an error and mark as invalid."""
        self.errors.append(message)
        self.is_valid = False

    def add_warning(self, message: str) -> None:
        """Add a warning (doesn't affect validity)."""
        self.warnings.append(message)

    def merge(self, other: "ValidationResult") -> "ValidationResult":
        """Merge with another validation result."""
        return ValidationResult(
            is_valid=self.is_valid and other.is_valid,
            errors=self.errors + other.errors,
            warnings=self.warnings + other.warnings,
            stats={**self.stats, **other.stats}
        )


@dataclass
class ColumnSpec:
    """
    Column specification for validation.

    Attributes:
        type: Expected data type
        nullable: Whether nulls are allowed
        unique: Whether values must be unique
        min: Minimum value (numeric)
        max: Maximum value (numeric)
        min_length: Minimum string length
        max_length: Maximum string length
        pattern: Regex pattern for strings
        allowed_values: Set of allowed values
        custom_check: Custom validation function
    """
    type: Optional[str] = None
    nullable: bool = True
    unique: bool = False
    min: Optional[float] = None
    max: Optional[float] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    pattern: Optional[str] = None
    allowed_values: Optional[List[Any]] = None
    custom_check: Optional[callable] = None


class ColumnValidator:
    """
    Validator for individual DataFrame columns.

    Example:
        >>> validator = ColumnValidator(
        ...     type="int",
        ...     nullable=False,
        ...     min=0,
        ...     max=100
        ... )
        >>> result = validator.validate(df["age"])
    """

    def __init__(
        self,
        type: Optional[str] = None,
        nullable: bool = True,
        unique: bool = False,
        min: Optional[float] = None,
        max: Optional[float] = None,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        pattern: Optional[str] = None,
        allowed_values: Optional[List[Any]] = None,
        custom_check: Optional[callable] = None
    ) -> None:
        """Initialize column validator with specifications."""
        self.spec = ColumnSpec(
            type=type,
            nullable=nullable,
            unique=unique,
            min=min,
            max=max,
            min_length=min_length,
            max_length=max_length,
            pattern=pattern,
            allowed_values=allowed_values,
            custom_check=custom_check
        )
        self._compiled_pattern: Optional[Pattern] = None
        if pattern:
            self._compiled_pattern = re.compile(pattern)

    def validate(self, series: Any, column_name: str = "column") -> ValidationResult:
        """
        Validate a column/series.

        Args:
            series: Column data (pandas Series, Spark Column, etc.)
            column_name: Name for error messages

        Returns:
            ValidationResult with validation outcome
        """
        result = ValidationResult()

        # Convert to pandas for validation if needed
        if hasattr(series, 'toPandas'):
            series = series.toPandas()
        elif hasattr(series, 'to_pandas'):
            series = series.to_pandas()

        # Null check
        if not self.spec.nullable:
            null_count = series.isna().sum()
            if null_count > 0:
                result.add_error(
                    f"Column '{column_name}' has {null_count} null values but nullable=False"
                )

        # Uniqueness check
        if self.spec.unique:
            duplicate_count = series.duplicated().sum()
            if duplicate_count > 0:
                result.add_error(
                    f"Column '{column_name}' has {duplicate_count} duplicate values but unique=True"
                )

        # Type check
        if self.spec.type:
            result = result.merge(
                self._validate_type(series, column_name)
            )

        # Range check (numeric)
        if self.spec.min is not None or self.spec.max is not None:
            result = result.merge(
                self._validate_range(series, column_name)
            )

        # Length check (strings)
        if self.spec.min_length is not None or self.spec.max_length is not None:
            result = result.merge(
                self._validate_length(series, column_name)
            )

        # Pattern check
        if self._compiled_pattern:
            result = result.merge(
                self._validate_pattern(series, column_name)
            )

        # Allowed values check
        if self.spec.allowed_values is not None:
            result = result.merge(
                self._validate_allowed_values(series, column_name)
            )

        # Custom check
        if self.spec.custom_check:
            try:
                if not self.spec.custom_check(series):
                    result.add_error(
                        f"Column '{column_name}' failed custom validation"
                    )
            except Exception as e:
                result.add_error(
                    f"Column '{column_name}' custom check raised exception: {e}"
                )

        return result

    def _validate_type(self, series: Any, column_name: str) -> ValidationResult:
        """Validate column data type."""
        result = ValidationResult()

        type_mapping = {
            "int": ["int", "int32", "int64", "Int32", "Int64"],
            "float": ["float", "float32", "float64"],
            "string": ["object", "string", "str"],
            "bool": ["bool", "boolean"],
            "datetime": ["datetime64", "datetime"],
        }

        expected_types = type_mapping.get(self.spec.type, [self.spec.type])
        actual_type = str(series.dtype)

        if not any(t in actual_type for t in expected_types):
            result.add_error(
                f"Column '{column_name}' has type '{actual_type}' "
                f"but expected '{self.spec.type}'"
            )

        return result

    def _validate_range(self, series: Any, column_name: str) -> ValidationResult:
        """Validate numeric range."""
        result = ValidationResult()
        non_null = series.dropna()

        if len(non_null) == 0:
            return result

        if self.spec.min is not None:
            below_min = (non_null < self.spec.min).sum()
            if below_min > 0:
                result.add_error(
                    f"Column '{column_name}' has {below_min} values below minimum {self.spec.min}"
                )

        if self.spec.max is not None:
            above_max = (non_null > self.spec.max).sum()
            if above_max > 0:
                result.add_error(
                    f"Column '{column_name}' has {above_max} values above maximum {self.spec.max}"
                )

        return result

    def _validate_length(self, series: Any, column_name: str) -> ValidationResult:
        """Validate string length."""
        result = ValidationResult()
        non_null = series.dropna()

        if len(non_null) == 0:
            return result

        lengths = non_null.astype(str).str.len()

        if self.spec.min_length is not None:
            too_short = (lengths < self.spec.min_length).sum()
            if too_short > 0:
                result.add_error(
                    f"Column '{column_name}' has {too_short} values shorter than {self.spec.min_length}"
                )

        if self.spec.max_length is not None:
            too_long = (lengths > self.spec.max_length).sum()
            if too_long > 0:
                result.add_error(
                    f"Column '{column_name}' has {too_long} values longer than {self.spec.max_length}"
                )

        return result

    def _validate_pattern(self, series: Any, column_name: str) -> ValidationResult:
        """Validate string pattern."""
        result = ValidationResult()
        non_null = series.dropna()

        if len(non_null) == 0:
            return result

        matches = non_null.astype(str).str.match(self.spec.pattern)
        non_matching = (~matches).sum()

        if non_matching > 0:
            result.add_error(
                f"Column '{column_name}' has {non_matching} values not matching pattern '{self.spec.pattern}'"
            )

        return result

    def _validate_allowed_values(self, series: Any, column_name: str) -> ValidationResult:
        """Validate against allowed values."""
        result = ValidationResult()
        non_null = series.dropna()

        if len(non_null) == 0:
            return result

        invalid = ~non_null.isin(self.spec.allowed_values)
        invalid_count = invalid.sum()

        if invalid_count > 0:
            invalid_values = non_null[invalid].unique()[:5]
            result.add_error(
                f"Column '{column_name}' has {invalid_count} invalid values. "
                f"Examples: {list(invalid_values)}"
            )

        return result


class SchemaValidator:
    """
    Validator for DataFrame schemas.

    Validates entire DataFrames against schema specifications.

    Example:
        >>> schema = {
        ...     "id": {"type": "int", "nullable": False},
        ...     "name": {"type": "string", "min_length": 1},
        ...     "amount": {"type": "float", "min": 0}
        ... }
        >>> validator = SchemaValidator(schema)
        >>> result = validator.validate(df)
    """

    def __init__(
        self,
        schema: Dict[str, Dict[str, Any]],
        strict: bool = False
    ) -> None:
        """
        Initialize schema validator.

        Args:
            schema: Schema specification dict
            strict: If True, fail on extra columns not in schema
        """
        self.schema = schema
        self.strict = strict
        self._column_validators: Dict[str, ColumnValidator] = {}

        # Create column validators
        for col_name, col_spec in schema.items():
            self._column_validators[col_name] = ColumnValidator(**col_spec)

    def validate(self, df: Any) -> ValidationResult:
        """
        Validate DataFrame against schema.

        Args:
            df: DataFrame to validate

        Returns:
            ValidationResult with validation outcome
        """
        result = ValidationResult()

        # Get column names
        if hasattr(df, 'columns'):
            df_columns = set(df.columns)
        else:
            df_columns = set()

        schema_columns = set(self.schema.keys())

        # Check for missing columns
        missing = schema_columns - df_columns
        if missing:
            result.add_error(f"Missing required columns: {missing}")

        # Check for extra columns (if strict)
        if self.strict:
            extra = df_columns - schema_columns
            if extra:
                result.add_error(f"Unexpected columns (strict mode): {extra}")

        # Validate each column
        for col_name, validator in self._column_validators.items():
            if col_name in df_columns:
                col_result = validator.validate(df[col_name], col_name)
                result = result.merge(col_result)

        return result

    def get_schema_info(self) -> Dict[str, Dict[str, Any]]:
        """Get schema information as dictionary."""
        return self.schema
