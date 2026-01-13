"""
DataForge Data Quality Example

This example demonstrates:
    - Schema validation
    - Column validation rules
    - Data profiling
    - Quality checks and assertions
    - Building data quality pipelines

Run this example:
    python examples/03_data_quality.py

Author: DataForge Team
"""

import sys
sys.path.insert(0, '.')

from dataforge import (
    DataFrame,
    SchemaValidator,
    ColumnValidator,
    DataProfiler,
)
from dataforge.quality.checks import (
    not_null,
    unique,
    in_range,
    in_set,
    matches_pattern,
    QualityCheck,
)
from dataforge.utils import setup_logging, get_logger

# Setup logging
setup_logging(level="INFO")
logger = get_logger(__name__)


def create_sample_data():
    """Create sample data with intentional quality issues."""
    return {
        "customer_id": [1, 2, 3, 4, 5, 6, None, 8, 9, 10],  # Has NULL
        "email": [
            "alice@example.com",
            "bob@example.com",
            "invalid-email",  # Invalid format
            "diana@example.com",
            "eve@example.com",
            "frank@example.com",
            "grace@example.com",
            "henry@example",  # Missing TLD
            "ivy@example.com",
            "jack@example.com",
        ],
        "age": [30, 25, 35, -5, 42, 28, 150, 33, 27, 45],  # Has negative and >120
        "status": [
            "active", "active", "inactive", "active", "pending",
            "invalid_status",  # Invalid value
            "active", "inactive", "active", "active"
        ],
        "revenue": [
            1000.0, 2500.0, 1500.0, 3000.0, None,  # Has NULL
            2000.0, 1800.0, 2200.0, 1600.0, 2800.0
        ],
        "signup_date": [
            "2024-01-15", "2024-02-20", "2024-03-10", "2024-04-05",
            "2024-05-12", "2024-06-18", "2024-07-22", "2024-08-30",
            "invalid-date",  # Invalid format
            "2024-10-11"
        ],
    }


def main():
    """Main entry point for data quality example."""

    print("=" * 70)
    print("DataForge Data Quality Example")
    print("=" * 70)

    # Create sample data with quality issues
    data = create_sample_data()
    df = DataFrame.from_dict(data, engine="pandas")

    print(f"\nSample data created: {df.count()} rows")

    # =========================================================================
    # 1. Data Profiling
    # =========================================================================
    print("\n1. DATA PROFILING")
    print("-" * 50)

    profiler = DataProfiler()
    profile_result = profiler.profile(df.to_pandas())

    print("\n  Column Profiles:")
    print("  " + "-" * 65)

    for col_name, col_profile in profile_result.columns.items():
        print(f"\n  {col_name}:")
        print(f"    Type: {col_profile.dtype}")
        print(f"    Non-null: {col_profile.non_null_count}/{col_profile.total_count}")
        print(f"    Null %: {col_profile.null_percentage:.1f}%")
        print(f"    Unique: {col_profile.unique_count}")

        if col_profile.dtype in ["int64", "float64"]:
            print(f"    Min: {col_profile.min_value}")
            print(f"    Max: {col_profile.max_value}")
            print(f"    Mean: {col_profile.mean:.2f}" if col_profile.mean else "")

    # =========================================================================
    # 2. Schema Validation
    # =========================================================================
    print("\n2. SCHEMA VALIDATION")
    print("-" * 50)

    expected_schema = {
        "customer_id": "int64",
        "email": "object",
        "age": "int64",
        "status": "object",
        "revenue": "float64",
        "signup_date": "object",
    }

    schema_validator = SchemaValidator(expected_schema)
    schema_result = schema_validator.validate(df.to_pandas())

    print(f"\n  Schema valid: {schema_result.is_valid}")
    if not schema_result.is_valid:
        print("  Errors:")
        for error in schema_result.errors:
            print(f"    - {error}")

    # =========================================================================
    # 3. Column Validation
    # =========================================================================
    print("\n3. COLUMN VALIDATION")
    print("-" * 50)

    # Define column validators
    validators = [
        ColumnValidator(
            column="customer_id",
            nullable=False,
            unique=True,
        ),
        ColumnValidator(
            column="email",
            nullable=False,
            pattern=r"^[\w\.-]+@[\w\.-]+\.\w+$",
        ),
        ColumnValidator(
            column="age",
            nullable=False,
            min_value=0,
            max_value=120,
        ),
        ColumnValidator(
            column="status",
            nullable=False,
            allowed_values=["active", "inactive", "pending"],
        ),
        ColumnValidator(
            column="revenue",
            nullable=True,  # Allow nulls
            min_value=0,
        ),
    ]

    print("\n  Validation Results:")
    print("  " + "-" * 50)

    pandas_df = df.to_pandas()
    all_valid = True

    for validator in validators:
        result = validator.validate(pandas_df)
        status = "PASS" if result.is_valid else "FAIL"
        print(f"\n  {validator.column}: {status}")

        if not result.is_valid:
            all_valid = False
            for error in result.errors[:3]:  # Show first 3 errors
                print(f"    - {error}")
            if len(result.errors) > 3:
                print(f"    ... and {len(result.errors) - 3} more errors")

    # =========================================================================
    # 4. Quality Checks
    # =========================================================================
    print("\n4. QUALITY CHECKS")
    print("-" * 50)

    # Define quality checks
    checks = [
        not_null("customer_id"),
        unique("customer_id"),
        in_range("age", 0, 120),
        in_set("status", ["active", "inactive", "pending"]),
        matches_pattern("email", r"^[\w\.-]+@[\w\.-]+\.\w+$"),
    ]

    print("\n  Running quality checks...")
    print("  " + "-" * 50)

    for check in checks:
        result = check.execute(pandas_df)
        status = "PASS" if result["passed"] else "FAIL"
        print(f"  {check.name}: {status}")
        if not result["passed"]:
            print(f"    Failed rows: {result.get('failed_count', 'N/A')}")

    # =========================================================================
    # 5. Quality Report
    # =========================================================================
    print("\n5. QUALITY REPORT SUMMARY")
    print("-" * 50)

    # Calculate quality metrics
    total_rows = len(pandas_df)
    total_cells = total_rows * len(pandas_df.columns)
    null_cells = pandas_df.isnull().sum().sum()
    completeness = ((total_cells - null_cells) / total_cells) * 100

    # Count issues
    issues = {
        "null_customer_id": pandas_df["customer_id"].isnull().sum(),
        "invalid_emails": sum(
            1 for email in pandas_df["email"]
            if not isinstance(email, str) or "@" not in email or "." not in email.split("@")[-1]
        ),
        "invalid_ages": sum(
            1 for age in pandas_df["age"]
            if age < 0 or age > 120
        ),
        "invalid_status": sum(
            1 for status in pandas_df["status"]
            if status not in ["active", "inactive", "pending"]
        ),
        "null_revenue": pandas_df["revenue"].isnull().sum(),
    }

    total_issues = sum(issues.values())

    print(f"""
    ┌────────────────────────────────────────────────────┐
    │              DATA QUALITY REPORT                    │
    ├────────────────────────────────────────────────────┤
    │  Total Rows:        {total_rows:<10}                      │
    │  Total Columns:     {len(pandas_df.columns):<10}                      │
    │  Completeness:      {completeness:.1f}%                        │
    │  Total Issues:      {total_issues:<10}                      │
    ├────────────────────────────────────────────────────┤
    │  Issue Breakdown:                                   │
    │    - Null customer_id:  {issues['null_customer_id']:<5}                     │
    │    - Invalid emails:    {issues['invalid_emails']:<5}                     │
    │    - Invalid ages:      {issues['invalid_ages']:<5}                     │
    │    - Invalid status:    {issues['invalid_status']:<5}                     │
    │    - Null revenue:      {issues['null_revenue']:<5}                     │
    └────────────────────────────────────────────────────┘
    """)

    # =========================================================================
    # 6. Building a Data Quality Pipeline
    # =========================================================================
    print("\n6. DATA QUALITY PIPELINE")
    print("-" * 50)

    print("""
    Best Practice: Build quality checks into your ETL pipeline

    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
    │   Ingest    │───▶│   Profile   │───▶│  Validate   │───▶│   Clean     │
    │   Data      │    │   Data      │    │   Rules     │    │   Data      │
    └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                                                │
                                                ▼
                                         ┌─────────────┐
                                         │   Alert on  │
                                         │   Failures  │
                                         └─────────────┘

    Pipeline Steps:
    1. Profile data to understand distributions
    2. Define validation rules based on business requirements
    3. Execute checks before and after transformations
    4. Log failures and optionally quarantine bad records
    5. Monitor data quality metrics over time
    """)

    # Example: Cleaning pipeline
    print("  Example: Cleaning the data...")

    # Remove rows with null customer_id
    clean_df = pandas_df[pandas_df["customer_id"].notna()].copy()

    # Fix invalid ages (clip to valid range)
    clean_df["age"] = clean_df["age"].clip(0, 120)

    # Fix invalid status
    clean_df.loc[
        ~clean_df["status"].isin(["active", "inactive", "pending"]),
        "status"
    ] = "pending"

    # Fill null revenue with median
    median_revenue = clean_df["revenue"].median()
    clean_df["revenue"] = clean_df["revenue"].fillna(median_revenue)

    print(f"  Original rows: {len(pandas_df)}")
    print(f"  Clean rows: {len(clean_df)}")
    print(f"  Removed: {len(pandas_df) - len(clean_df)} rows")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print("""
    1. PROFILE FIRST: Understand your data before defining rules
       - Identify data types and distributions
       - Find outliers and anomalies
       - Measure completeness

    2. DEFINE CLEAR RULES: Based on business requirements
       - Not null constraints for required fields
       - Range checks for numeric values
       - Pattern matching for formatted data (email, phone)
       - Referential integrity for foreign keys

    3. VALIDATE AT BOUNDARIES: Check data quality at:
       - Data ingestion (source validation)
       - After transformations (pipeline validation)
       - Before publishing (output validation)

    4. HANDLE FAILURES GRACEFULLY:
       - Log validation errors
       - Quarantine bad records (don't lose data)
       - Alert on threshold breaches
       - Provide clear error messages

    5. MONITOR OVER TIME:
       - Track quality metrics trends
       - Set up alerts for degradation
       - Document data quality SLAs
    """)

    print("Data quality example completed!")


if __name__ == "__main__":
    main()
