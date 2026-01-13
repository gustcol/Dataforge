"""
DataForge Exception Hierarchy

This module defines the custom exception hierarchy used throughout the DataForge
framework. All exceptions inherit from DataForgeError for easy catch-all handling.

Exception Hierarchy:
    DataForgeError (base)
    ├── EngineNotAvailableError - Requested engine is not installed/available
    ├── DataSizeExceededError - Data size exceeds engine capacity
    ├── ConfigurationError - Invalid configuration parameters
    ├── ValidationError - Data validation failures
    ├── TransformationError - Errors during data transformation
    └── StreamingError - Errors in streaming operations

Example:
    >>> from dataforge.core.exceptions import DataForgeError, EngineNotAvailableError
    >>>
    >>> try:
    ...     engine = get_rapids_engine()
    ... except EngineNotAvailableError as e:
    ...     print(f"RAPIDS not available: {e}")
    ...     # Fall back to pandas
    ...     engine = get_pandas_engine()
"""

from typing import Optional, Any, Dict, List


class DataForgeError(Exception):
    """
    Base exception for all DataForge errors.

    All custom exceptions in DataForge inherit from this class, allowing
    users to catch all framework-specific errors with a single except clause.

    Attributes:
        message: Human-readable error description
        details: Optional dictionary with additional error context
        original_error: Original exception if this wraps another error

    Example:
        >>> try:
        ...     # DataForge operations
        ...     pass
        ... except DataForgeError as e:
        ...     logger.error(f"DataForge error: {e}")
        ...     if e.details:
        ...         logger.debug(f"Details: {e.details}")
    """

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None
    ) -> None:
        """
        Initialize DataForgeError.

        Args:
            message: Human-readable error description
            details: Optional dictionary with additional context
            original_error: Original exception if wrapping another error
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.original_error = original_error

    def __str__(self) -> str:
        """Return string representation of the error."""
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message

    def __repr__(self) -> str:
        """Return detailed representation for debugging."""
        return (
            f"{self.__class__.__name__}("
            f"message={self.message!r}, "
            f"details={self.details!r}, "
            f"original_error={self.original_error!r})"
        )


class EngineNotAvailableError(DataForgeError):
    """
    Raised when a requested engine is not available.

    This error occurs when:
        - The required library is not installed (e.g., pyspark, cudf)
        - Required hardware is not available (e.g., no GPU for RAPIDS)
        - The engine cannot be initialized due to environment issues

    Attributes:
        engine_name: Name of the unavailable engine
        install_hint: Suggestion for how to install the required dependencies

    Example:
        >>> try:
        ...     from dataforge.engines import RapidsEngine
        ...     engine = RapidsEngine()
        ... except EngineNotAvailableError as e:
        ...     print(f"Cannot use {e.engine_name}: {e.message}")
        ...     print(f"Install hint: {e.install_hint}")
    """

    def __init__(
        self,
        engine_name: str,
        message: Optional[str] = None,
        install_hint: Optional[str] = None,
        original_error: Optional[Exception] = None
    ) -> None:
        """
        Initialize EngineNotAvailableError.

        Args:
            engine_name: Name of the unavailable engine ('pandas', 'spark', 'rapids')
            message: Custom error message (auto-generated if not provided)
            install_hint: Installation instructions for the missing dependency
            original_error: Original exception if wrapping another error
        """
        self.engine_name = engine_name
        self.install_hint = install_hint or self._get_default_install_hint(engine_name)

        if message is None:
            message = f"Engine '{engine_name}' is not available"

        super().__init__(
            message=message,
            details={
                "engine_name": engine_name,
                "install_hint": self.install_hint
            },
            original_error=original_error
        )

    @staticmethod
    def _get_default_install_hint(engine_name: str) -> str:
        """Get default installation hint for an engine."""
        hints = {
            "pandas": "pip install pandas",
            "spark": "pip install pyspark",
            "rapids": "pip install cudf-cu12 --extra-index-url=https://pypi.nvidia.com",
        }
        return hints.get(engine_name.lower(), f"pip install {engine_name}")


class DataSizeExceededError(DataForgeError):
    """
    Raised when data size exceeds the capacity of the selected engine.

    This error helps users understand when they need to switch to a more
    scalable engine or implement chunked processing.

    Attributes:
        actual_size_bytes: Actual size of the data in bytes
        max_size_bytes: Maximum size the engine can handle
        recommended_engine: Suggested alternative engine

    Example:
        >>> try:
        ...     df = pandas_engine.read_csv("huge_file.csv")
        ... except DataSizeExceededError as e:
        ...     print(f"Data too large: {e.actual_size_bytes / 1e9:.2f} GB")
        ...     print(f"Recommended: {e.recommended_engine}")
    """

    def __init__(
        self,
        actual_size_bytes: int,
        max_size_bytes: int,
        recommended_engine: Optional[str] = None,
        message: Optional[str] = None
    ) -> None:
        """
        Initialize DataSizeExceededError.

        Args:
            actual_size_bytes: Actual size of the data in bytes
            max_size_bytes: Maximum size the engine can handle
            recommended_engine: Suggested alternative engine
            message: Custom error message
        """
        self.actual_size_bytes = actual_size_bytes
        self.max_size_bytes = max_size_bytes
        self.recommended_engine = recommended_engine

        actual_gb = actual_size_bytes / (1024 ** 3)
        max_gb = max_size_bytes / (1024 ** 3)

        if message is None:
            message = (
                f"Data size ({actual_gb:.2f} GB) exceeds maximum "
                f"capacity ({max_gb:.2f} GB)"
            )
            if recommended_engine:
                message += f". Consider using '{recommended_engine}' engine."

        super().__init__(
            message=message,
            details={
                "actual_size_bytes": actual_size_bytes,
                "max_size_bytes": max_size_bytes,
                "actual_size_gb": actual_gb,
                "max_size_gb": max_gb,
                "recommended_engine": recommended_engine
            }
        )


class ConfigurationError(DataForgeError):
    """
    Raised when configuration parameters are invalid.

    This error occurs when:
        - Required configuration values are missing
        - Configuration values are out of valid range
        - Incompatible configuration options are combined

    Attributes:
        parameter_name: Name of the invalid parameter
        provided_value: The invalid value that was provided
        valid_values: List or description of valid values

    Example:
        >>> try:
        ...     config = SparkConfig(shuffle_partitions=-1)
        ... except ConfigurationError as e:
        ...     print(f"Invalid config for '{e.parameter_name}'")
        ...     print(f"Valid values: {e.valid_values}")
    """

    def __init__(
        self,
        parameter_name: str,
        message: Optional[str] = None,
        provided_value: Any = None,
        valid_values: Optional[Any] = None
    ) -> None:
        """
        Initialize ConfigurationError.

        Args:
            parameter_name: Name of the invalid parameter
            message: Custom error message
            provided_value: The invalid value that was provided
            valid_values: Description or list of valid values
        """
        self.parameter_name = parameter_name
        self.provided_value = provided_value
        self.valid_values = valid_values

        if message is None:
            message = f"Invalid configuration for parameter '{parameter_name}'"
            if provided_value is not None:
                message += f": got {provided_value!r}"
            if valid_values is not None:
                message += f" (valid: {valid_values})"

        super().__init__(
            message=message,
            details={
                "parameter_name": parameter_name,
                "provided_value": provided_value,
                "valid_values": valid_values
            }
        )


class ValidationError(DataForgeError):
    """
    Raised when data validation fails.

    This error is used by the quality module to report validation failures
    including schema mismatches, constraint violations, and data quality issues.

    Attributes:
        validation_type: Type of validation that failed
        column_name: Name of the column that failed validation (if applicable)
        failures: List of specific validation failures

    Example:
        >>> try:
        ...     validator.validate(df)
        ... except ValidationError as e:
        ...     for failure in e.failures:
        ...         print(f"Column '{failure['column']}': {failure['error']}")
    """

    def __init__(
        self,
        validation_type: str,
        message: Optional[str] = None,
        column_name: Optional[str] = None,
        failures: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """
        Initialize ValidationError.

        Args:
            validation_type: Type of validation ('schema', 'constraint', 'quality')
            message: Custom error message
            column_name: Specific column that failed (if applicable)
            failures: List of detailed failure information
        """
        self.validation_type = validation_type
        self.column_name = column_name
        self.failures = failures or []

        if message is None:
            message = f"Data validation failed: {validation_type}"
            if column_name:
                message += f" (column: {column_name})"
            if failures:
                message += f" - {len(failures)} failure(s)"

        super().__init__(
            message=message,
            details={
                "validation_type": validation_type,
                "column_name": column_name,
                "failures": failures,
                "failure_count": len(failures) if failures else 0
            }
        )


class TransformationError(DataForgeError):
    """
    Raised when a data transformation operation fails.

    This error occurs during operations like:
        - Filter operations with invalid expressions
        - Join operations with incompatible schemas
        - Aggregation operations with unsupported functions

    Attributes:
        operation: Name of the failed operation
        expression: The expression or parameters that caused the failure

    Example:
        >>> try:
        ...     df.filter("invalid_column > 10")
        ... except TransformationError as e:
        ...     print(f"Operation '{e.operation}' failed")
        ...     print(f"Expression: {e.expression}")
    """

    def __init__(
        self,
        operation: str,
        message: Optional[str] = None,
        expression: Optional[str] = None,
        original_error: Optional[Exception] = None
    ) -> None:
        """
        Initialize TransformationError.

        Args:
            operation: Name of the transformation operation
            message: Custom error message
            expression: The expression that caused the failure
            original_error: Original exception if wrapping another error
        """
        self.operation = operation
        self.expression = expression

        if message is None:
            message = f"Transformation operation '{operation}' failed"
            if expression:
                message += f": {expression}"

        super().__init__(
            message=message,
            details={
                "operation": operation,
                "expression": expression
            },
            original_error=original_error
        )


class StreamingError(DataForgeError):
    """
    Raised when a streaming operation fails.

    This error is specific to Spark Structured Streaming operations
    and includes information about the streaming query state.

    Attributes:
        query_name: Name of the streaming query
        query_id: Unique identifier of the query
        state: Current state of the query when error occurred

    Example:
        >>> try:
        ...     stream.start()
        ... except StreamingError as e:
        ...     print(f"Stream '{e.query_name}' failed in state: {e.state}")
    """

    def __init__(
        self,
        query_name: str,
        message: Optional[str] = None,
        query_id: Optional[str] = None,
        state: Optional[str] = None,
        original_error: Optional[Exception] = None
    ) -> None:
        """
        Initialize StreamingError.

        Args:
            query_name: Name of the streaming query
            message: Custom error message
            query_id: Unique identifier of the query
            state: Current state when error occurred
            original_error: Original exception if wrapping another error
        """
        self.query_name = query_name
        self.query_id = query_id
        self.state = state

        if message is None:
            message = f"Streaming query '{query_name}' failed"
            if state:
                message += f" (state: {state})"

        super().__init__(
            message=message,
            details={
                "query_name": query_name,
                "query_id": query_id,
                "state": state
            },
            original_error=original_error
        )


class DatabricksError(DataForgeError):
    """
    Raised when a Databricks-specific operation fails.

    This error is used for Delta Lake, Unity Catalog, and other
    Databricks-specific functionality.

    Attributes:
        component: Databricks component that failed ('delta', 'unity_catalog', 'photon')
        resource: Specific resource that caused the failure (table name, catalog, etc.)

    Example:
        >>> try:
        ...     delta_manager.optimize("my_table")
        ... except DatabricksError as e:
        ...     print(f"Databricks {e.component} error: {e.message}")
    """

    def __init__(
        self,
        component: str,
        message: Optional[str] = None,
        resource: Optional[str] = None,
        original_error: Optional[Exception] = None
    ) -> None:
        """
        Initialize DatabricksError.

        Args:
            component: Databricks component ('delta', 'unity_catalog', 'photon')
            message: Custom error message
            resource: Specific resource involved
            original_error: Original exception if wrapping another error
        """
        self.component = component
        self.resource = resource

        if message is None:
            message = f"Databricks {component} operation failed"
            if resource:
                message += f" for resource '{resource}'"

        super().__init__(
            message=message,
            details={
                "component": component,
                "resource": resource
            },
            original_error=original_error
        )
