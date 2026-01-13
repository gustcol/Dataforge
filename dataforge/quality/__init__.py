"""
DataForge Quality Module

This module provides data quality validation and profiling utilities.

Components:
    - SchemaValidator: Validate DataFrame schemas
    - ColumnValidator: Validate individual columns
    - DataProfiler: Generate data quality profiles
    - QualityChecks: Pre-defined quality checks

Example:
    >>> from dataforge.quality import SchemaValidator, DataProfiler
    >>>
    >>> # Validate schema
    >>> validator = SchemaValidator({
    ...     "id": {"type": "int", "nullable": False},
    ...     "email": {"type": "string", "nullable": False, "pattern": r".*@.*"},
    ...     "amount": {"type": "float", "min": 0}
    ... })
    >>> validator.validate(df)
    >>>
    >>> # Profile data
    >>> profiler = DataProfiler()
    >>> profile = profiler.profile(df)
    >>> print(profile.summary())
"""

from dataforge.quality.validators import SchemaValidator, ColumnValidator
from dataforge.quality.profiler import DataProfiler
from dataforge.quality.checks import QualityCheck, run_quality_checks

__all__ = [
    "SchemaValidator",
    "ColumnValidator",
    "DataProfiler",
    "QualityCheck",
    "run_quality_checks",
]
