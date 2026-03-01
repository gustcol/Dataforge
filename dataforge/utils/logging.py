"""
DataForge Logging Utilities

Centralized logging configuration for the DataForge framework.

Features:
    - Structured logging
    - Multiple output formats
    - Level configuration
    - Context injection

Example:
    >>> from dataforge.utils import setup_logging, get_logger
    >>>
    >>> # Setup logging
    >>> setup_logging(level="INFO", format="json")
    >>>
    >>> # Get logger
    >>> logger = get_logger("my_module")
    >>> logger.info("Processing started", extra={"rows": 1000})
"""

from dataclasses import dataclass
from typing import Optional
import logging
import sys
import json
from datetime import datetime, timezone


@dataclass
class LogConfig:
    """Logging configuration.

    Attributes:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format: Output format ("text", "json")
        include_timestamp: Include timestamp in output
        include_module: Include module name in output
        log_file: Optional file path for log output
    """
    level: str = "INFO"
    format: str = "text"
    include_timestamp: bool = True
    include_module: bool = True
    log_file: Optional[str] = None


class JsonFormatter(logging.Formatter):
    """JSON log formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add extra fields
        if hasattr(record, '__dict__'):
            for key, value in record.__dict__.items():
                if key not in (
                    'name', 'msg', 'args', 'created', 'filename',
                    'funcName', 'levelname', 'levelno', 'lineno',
                    'module', 'msecs', 'pathname', 'process',
                    'processName', 'relativeCreated', 'stack_info',
                    'exc_info', 'exc_text', 'thread', 'threadName',
                    'message'
                ):
                    log_data[key] = value

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data)


class TextFormatter(logging.Formatter):
    """Standard text formatter with customization."""

    def __init__(
        self,
        include_timestamp: bool = True,
        include_module: bool = True
    ):
        parts = []

        if include_timestamp:
            parts.append("%(asctime)s")

        parts.append("%(levelname)-8s")

        if include_module:
            parts.append("[%(name)s]")

        parts.append("%(message)s")

        fmt = " ".join(parts)
        super().__init__(fmt, datefmt="%Y-%m-%d %H:%M:%S")


def setup_logging(
    level: str = "INFO",
    format: str = "text",
    include_timestamp: bool = True,
    include_module: bool = True,
    log_file: Optional[str] = None,
    root_logger: bool = True
) -> None:
    """
    Configure logging for DataForge.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format: Output format ("text" or "json")
        include_timestamp: Include timestamp in text format
        include_module: Include module name in text format
        log_file: Optional file path for log output
        root_logger: Configure root logger (affects all loggers)

    Example:
        >>> # Basic setup
        >>> setup_logging(level="INFO")
        >>>
        >>> # JSON format for production
        >>> setup_logging(level="WARNING", format="json", log_file="app.log")
        >>>
        >>> # Debug mode
        >>> setup_logging(level="DEBUG", include_timestamp=False)
    """
    # Determine logger to configure
    logger = logging.getLogger() if root_logger else logging.getLogger("dataforge")

    # Set level
    log_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(log_level)

    # Clear existing handlers
    logger.handlers.clear()

    # Create formatter
    if format.lower() == "json":
        formatter = JsonFormatter()
    else:
        formatter = TextFormatter(
            include_timestamp=include_timestamp,
            include_module=include_module
        )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Reduce noise from other libraries
    logging.getLogger("py4j").setLevel(logging.WARNING)
    logging.getLogger("pyspark").setLevel(logging.WARNING)

    logger.debug(f"Logging configured: level={level}, format={format}")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance.

    Args:
        name: Logger name (typically module name)

    Returns:
        Logger instance

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Starting processing")
    """
    return logging.getLogger(f"dataforge.{name}")


class LogContext:
    """
    Context manager for adding context to log messages.

    Example:
        >>> logger = get_logger("processor")
        >>>
        >>> with LogContext(logger, job_id="123", table="sales"):
        ...     logger.info("Processing started")
        ...     # Logs: "Processing started" with job_id=123, table=sales
    """

    def __init__(self, logger: logging.Logger, **context):
        """
        Initialize log context.

        Args:
            logger: Logger to add context to
            **context: Context key-value pairs
        """
        self.logger = logger
        self.context = context
        self._original_factory = None

    def __enter__(self):
        """Add context to log records."""
        old_factory = logging.getLogRecordFactory()
        context = self.context

        def record_factory(*args, **kwargs):
            record = old_factory(*args, **kwargs)
            for key, value in context.items():
                setattr(record, key, value)
            return record

        self._original_factory = old_factory
        logging.setLogRecordFactory(record_factory)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore original log record factory."""
        if self._original_factory:
            logging.setLogRecordFactory(self._original_factory)


def log_execution_time(logger: logging.Logger, operation: str):
    """
    Decorator for logging function execution time.

    Args:
        logger: Logger instance
        operation: Operation name for logging

    Example:
        >>> logger = get_logger("pipeline")
        >>>
        >>> @log_execution_time(logger, "data_transform")
        >>> def transform_data(df):
        ...     return df.groupby("col").sum()
    """
    import time
    from functools import wraps

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                duration = time.perf_counter() - start
                logger.info(
                    f"{operation} completed",
                    extra={"duration_seconds": duration, "status": "success"}
                )
                return result
            except Exception as e:
                duration = time.perf_counter() - start
                logger.error(
                    f"{operation} failed: {e}",
                    extra={"duration_seconds": duration, "status": "error"}
                )
                raise

        return wrapper
    return decorator
