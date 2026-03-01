# DataForge Testing Documentation

This document describes the testing infrastructure, tools, and procedures for the DataForge framework.

## Table of Contents

- [Testing Tools](#testing-tools)
- [Python Testing](#python-testing)
- [Scala Testing](#scala-testing)
- [S3 Integration Testing](#s3-integration-testing)
- [Running All Tests](#running-all-tests)
- [Test Results Summary](#test-results-summary)

---

## Testing Tools

### Python Tools

| Tool | Purpose | Installation |
|------|---------|--------------|
| **pytest** | Unit and integration testing | `pip install pytest` |
| **ty** | Type checking (astral-sh/ty) | `uv tool install ty` or `pip install ty` |
| **ruff** | Fast linting and formatting | `pip install ruff` |
| **pre-commit** | Git hook manager | `pip install pre-commit` |
| **moto** | AWS mock for S3 testing | `pip install moto[s3]` |
| **boto3** | AWS SDK for S3 tests | `pip install boto3` |

### Shell Tools

| Tool | Purpose | Installation |
|------|---------|--------------|
| **shellcheck** | Shell script static analysis | `brew install shellcheck` or `apt install shellcheck` |

### Scala Tools

| Tool | Purpose | Location |
|------|---------|----------|
| **scalint** | Scala static analysis | `/Users/gustcol/Pessoal/scalint/target/scala-2.13/scalint.jar` |

### Infrastructure Tools

| Tool | Purpose | Installation |
|------|---------|--------------|
| **gofakes3** | Fake S3 server for testing | `go install github.com/johannesboyne/gofakes3/cmd/gofakes3@latest` |
| **uv/uvx** | Fast Python package management | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |

---

## Python Testing

### Unit Tests

Run unit tests with pytest:

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_core.py -v

# Run with coverage
python -m pytest tests/ --cov=dataforge --cov-report=html
```

### Type Checking with ty

[ty](https://github.com/astral-sh/ty) is a fast Python type checker from astral-sh (makers of ruff).

```bash
# Check all Python code
ty check dataforge/

# Check specific module
ty check dataforge/core/

# With custom environment
ty check dataforge/ --python-version 3.10
```

#### Current ty Results

```
Total diagnostics: 156
- Import errors (expected): 88 (optional dependencies not installed)
- Code errors: 67 (pre-existing patterns in engine implementations)
- Warnings: 1
```

Most code errors are due to:
- Type inference with heterogeneous dictionaries
- Optional dependency type stubs not available (pyspark, cudf, etc.)
- Strict method override checking in streaming sinks
- Union type narrowing for conditional imports

The new Polars engine (`polars_engine.py`) passes ty with zero errors.

#### ty Configuration

Configuration is in `pyproject.toml`:

```toml
[tool.ty.environment]
python-version = "3.10"

[tool.ty.rules]
# ty is a type checker focused on detecting type errors.
# External libraries (pyspark, cudf, etc.) may not have complete type stubs.
```

### Installing Python Dependencies

Using uv (recommended):

```bash
# Create virtual environment
uv venv

# Install dependencies
uv pip install -e ".[dev]"

# Run tests
uv run pytest tests/ -v
```

Using pip:

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v
```

---

## Scala Testing

### Static Analysis with scalint

[scalint](https://github.com/gustcol/scalint) is a comprehensive Scala static analysis tool with 60+ rules.

```bash
# Run on all Scala files
java -jar /path/to/scalint.jar scala/src/main/scala

# Output as JSON
java -jar /path/to/scalint.jar --format json scala/src/main/scala

# Only security checks
java -jar /path/to/scalint.jar --category security scala/src/main/scala

# Fail on warnings (for CI)
java -jar /path/to/scalint.jar --fail-on-warning scala/src/main/scala
```

#### Current scalint Results

```
Files analyzed: 5/5
Issues found:
- Errors: 0
- Warnings: 56
- Info: 23
- Hints: 1
```

#### Common Warnings

| Rule | Count | Description |
|------|-------|-------------|
| S004 | 24 | Variable naming (constants should be UPPER_SNAKE_CASE) |
| F002 | 12 | Side effects in foreach operations |
| C001 | 5 | Mutable variables in concurrent context |
| S011 | 10 | Wildcard imports |
| F004 | 8 | Accumulator patterns (consider fold/reduce) |
| B003 | 3 | Using .head on collections |

#### scalint Rule Categories

- **Style** (S001-S011): Naming conventions, code style
- **Bug** (B001-B013): Potential bugs, null usage
- **Performance** (P001-P010): Efficiency issues
- **Security** (SEC001-SEC008): Security vulnerabilities
- **Concurrency** (C001-C008): Thread safety
- **Functional** (F001-F010): FP best practices

---

## S3 Integration Testing

### Testing Modes

The S3 integration tests support two modes:

#### 1. Mock Mode (Default) - Uses moto

```bash
# Run with moto mock (fast, isolated)
python -m pytest tests/test_s3_integration.py -v
```

#### 2. Fake S3 Mode - Uses gofakes3

```bash
# Start gofakes3 server
gofakes3 -backend memory -autobucket &

# Run tests with fake S3
S3_ENDPOINT=http://localhost:9000 python -m pytest tests/test_s3_integration.py -v

# Stop the server
pkill gofakes3
```

### Installing gofakes3

```bash
# Requires Go 1.21+
go install github.com/johannesboyne/gofakes3/cmd/gofakes3@latest

# Verify installation
~/go/bin/gofakes3 --help
```

### Test Classes

| Test Class | Purpose |
|------------|---------|
| `TestS3Config` | S3 configuration validation |
| `TestS3Optimizer` | S3 optimizer functionality |
| `TestS3Integration` | S3 upload/download operations |
| `TestFormatAdvisor` | File format recommendations |
| `TestStorageAnalyzer` | Storage analysis utilities |

### S3 Test Results

```
tests/test_s3_integration.py: 18 tests passed
```

---

## Running All Tests

### Quick Test (Python only)

```bash
# Unit tests
python -m pytest tests/test_core.py -v

# S3 integration tests
python -m pytest tests/test_s3_integration.py -v
```

### Full Test Suite

```bash
#!/bin/bash

echo "=== DataForge Test Suite ==="

# 1. Python Unit Tests
echo "\n--- Python Unit Tests ---"
python -m pytest tests/test_core.py -v

# 2. Python S3 Integration Tests
echo "\n--- S3 Integration Tests ---"
python -m pytest tests/test_s3_integration.py -v

# 3. Python Type Checking
echo "\n--- Python Type Checking (ty) ---"
ty check dataforge/ 2>&1 | grep -v "unresolved-import" | tail -10

# 4. Scala Static Analysis
echo "\n--- Scala Static Analysis (scalint) ---"
java -jar /Users/gustcol/Pessoal/scalint/target/scala-2.13/scalint.jar \
    --severity warning \
    scala/src/main/scala 2>&1 | tail -20

echo "\n=== Test Suite Complete ==="
```

### CI/CD Integration

#### GitHub Actions Example

```yaml
name: DataForge Tests

on: [push, pull_request]

jobs:
  python-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: pip install -e ".[dev]"

      - name: Run pytest
        run: pytest tests/ -v

      - name: Run ty
        run: |
          pip install ty
          ty check dataforge/ || true

  scala-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up JDK
        uses: actions/setup-java@v4
        with:
          java-version: '17'
          distribution: 'temurin'

      - name: Run scalint
        run: |
          # Download or build scalint
          java -jar scalint.jar --format github --severity warning scala/src/main/scala
```

---

## Test Results Summary

### Python Tests

| Test File | Tests | Passed | Failed |
|-----------|-------|--------|--------|
| test_core.py | 26 | 26 | 0 |
| test_imports.py | 36 | 36 | 0 |
| test_s3_integration.py | 18 | 18 | 0 |
| **Total** | **80** | **80** | **0** |

### Python Type Checking (ty)

| Category | Count |
|----------|-------|
| Unresolved imports | 88 (expected - optional deps not installed) |
| Code errors | 67 (pre-existing, mostly optional dep patterns) |
| Warnings | 1 |
| **Total diagnostics** | **156** |

Note: The Polars engine (`dataforge/engines/polars_engine.py`) has **zero** ty errors.

### Scala Static Analysis (scalint)

| Severity | Count |
|----------|-------|
| Errors | 0 |
| Warnings | 56 |
| Info | 23 |
| Hints | 1 |
| **Total issues** | **80** |

---

## Recommendations

### Python

1. Install optional dependencies (pyspark, cudf) to reduce ty import errors
2. Consider adding type stubs for external dependencies
3. Review method signatures in streaming/sinks.py for stricter type compliance

### Scala

1. Fix constant naming to use UPPER_SNAKE_CASE
2. Replace `.head` calls with `.headOption`
3. Add `@volatile` annotation to mutable variables in concurrent code
4. Consider using `foldLeft` instead of var + foreach pattern
5. Replace wildcard imports with specific imports

---

## Troubleshooting

### ty shows many import errors

This is expected for optional dependencies. Run:
```bash
ty check dataforge/ 2>&1 | grep -v "unresolved-import"
```

### gofakes3 not starting

Ensure Go is installed and PATH includes `~/go/bin`:
```bash
export PATH=$PATH:~/go/bin
gofakes3 -backend memory -autobucket
```

### scalint not found

Build from source:
```bash
cd /Users/gustcol/Pessoal/scalint
sbt assembly
```

### Tests fail with import errors

Install dev dependencies:
```bash
pip install -e ".[dev]"
```
