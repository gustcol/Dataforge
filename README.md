# DataForge

**Intelligent Data Processing Framework for Pandas, Spark, and RAPIDS**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

DataForge is a production-ready Python framework that intelligently selects and utilizes the optimal data processing engine (Pandas, Polars, PySpark, or RAPIDS/cuDF) based on dataset characteristics, available hardware, and use case requirements.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Engine Selection](#engine-selection)
- [Unified API](#unified-api)
- [Data Quality](#data-quality)
- [ML Pipeline](#ml-pipeline)
- [Streaming](#streaming)
- [Databricks Integration](#databricks-integration)
  - [Delta Lake](#delta-lake)
  - [Unity Catalog](#unity-catalog)
  - [Photon](#photon)
- [S3 Storage Optimization](#s3-storage-optimization)
- [Scala Library](#scala-library)
- [Best Practices](#best-practices)
- [API Reference](#api-reference)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Unified API**: Single interface for Pandas, Polars, Spark, and RAPIDS DataFrames
- **Automatic Engine Selection**: Intelligent recommendation based on data size and hardware
- **Polars Integration**: High-performance Rust-backed engine for medium datasets
- **Full Databricks Integration**: Delta Lake, Unity Catalog, Photon optimization
- **S3 Storage Optimization**: File compaction, format conversion, cost analysis
- **Data Quality**: Schema validation, profiling, and quality checks
- **ML Pipeline**: Feature engineering and MLflow integration
- **Structured Streaming**: Sources, sinks, and processing patterns
- **Benchmarking**: Performance profiling across engines

## Installation

### Basic Installation

```bash
pip install dataforge
```

### With Optional Dependencies

```bash
# With Polars support (recommended for single-node workloads)
pip install dataforge[polars]

# With Spark support
pip install dataforge[spark]

# With ML support
pip install dataforge[ml]

# With all features
pip install dataforge[all]

# Development installation
pip install dataforge[dev]
```

### From Source

```bash
git clone https://github.com/dataforge/dataforge.git
cd dataforge
pip install -e ".[all]"
```

## Quick Start

```python
from dataforge import DataFrame, EngineRecommender

# Get engine recommendation for your data
recommender = EngineRecommender()
recommendation = recommender.recommend(data_size_gb=5.0)
print(recommendation)
# EngineRecommendation(engine='spark', reason='Dataset > 1GB requires distributed processing')

# Use the unified DataFrame API
df = DataFrame.read_csv("data.csv", engine="auto")
result = df.filter("age > 30").groupby("city").agg({"salary": "mean"})
result.write_parquet("output.parquet")
```

## Engine Selection

DataForge automatically recommends the optimal engine based on your data and hardware:

### Selection Matrix

| Dataset Size | Hardware | Recommended Engine | Performance Gain |
|-------------|----------|-------------------|------------------|
| < 100 MB | Any | **Pandas** | Baseline |
| 100 MB - 1 GB | No GPU | **Polars** | 5-20x vs Pandas |
| 100 MB - 1 GB | GPU | **RAPIDS** | 5-20x vs Pandas |
| 1 GB - 10 GB | No GPU, No Cluster | **Polars** | Best single-node performance |
| 1 GB - 10 GB | With Cluster | **Spark** | Distributed processing |
| 1 GB - 10 GB | GPU | **RAPIDS** | 20-100x vs Pandas |
| > 10 GB | Any | **Spark** | Distributed required |

### Using the Recommender

```python
from dataforge import EngineRecommender
from dataforge.advisor.hardware_detector import HardwareDetector

# Detect available hardware
detector = HardwareDetector()
capabilities = detector.detect()
print(f"GPU Available: {capabilities.gpu_available}")
print(f"Spark Available: {capabilities.spark_available}")

# Get recommendation
recommender = EngineRecommender()
rec = recommender.recommend(
    data_size_gb=5.0,
    has_gpu=capabilities.gpu_available,
    operation_type="aggregation"
)
print(f"Recommended: {rec.engine.value}")
print(f"Reason: {rec.reason}")
```

## Unified API

DataForge provides a unified API that works across all engines:

```python
from dataforge import DataFrame

# Create DataFrame (auto-selects engine based on size)
df = DataFrame.from_dict({
    "id": [1, 2, 3, 4, 5],
    "name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
    "age": [30, 25, 35, 28, 32],
    "salary": [75000, 65000, 85000, 70000, 80000],
}, engine="auto")

# Transformations (same API regardless of engine)
result = (
    df
    .filter("age > 25")
    .select(["name", "salary"])
    .with_column("bonus", "salary * 0.1")
    .sort("salary", ascending=False)
)

# Access native DataFrame when needed
pandas_df = result.to_pandas()
spark_df = result.to_spark(spark_session)
```

### Polars Engine

Polars is a high-performance DataFrame library written in Rust that provides lazy evaluation and parallel execution:

```python
from dataforge.engines import PolarsEngine
from dataforge.core.config import PolarsConfig

# Configure Polars engine
config = PolarsConfig(use_lazy=True, streaming=True)
engine = PolarsEngine(config)

# Read and process data
df = engine.read_parquet("data.parquet")
result = engine.filter(df, "age > 30")
result = engine.groupby(result, ["city"], {"salary": ["mean", "sum"]})

# Convert to pandas when needed
pandas_df = engine.to_pandas(result)
```

### Engine Escape Hatches

```python
# Access native engine for engine-specific operations
native_df = df.native  # Returns underlying Pandas/Polars/Spark/cuDF DataFrame

# Convert between engines
pandas_df = df.to_pandas()
spark_df = df.to_spark(spark)
cudf_df = df.to_cudf()
```

## Data Quality

### Schema Validation

```python
from dataforge import SchemaValidator, ColumnValidator

# Define expected schema
schema_validator = SchemaValidator({
    "customer_id": "int64",
    "email": "object",
    "age": "int64",
})

# Validate
result = schema_validator.validate(df.to_pandas())
if not result.is_valid:
    print(f"Schema errors: {result.errors}")
```

### Column Validation

```python
# Define column rules
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
]

# Validate each column
for validator in validators:
    result = validator.validate(df.to_pandas())
    print(f"{validator.column}: {'PASS' if result.is_valid else 'FAIL'}")
```

### Data Profiling

```python
from dataforge import DataProfiler

profiler = DataProfiler()
profile = profiler.profile(df.to_pandas())

for col_name, col_profile in profile.columns.items():
    print(f"{col_name}:")
    print(f"  Type: {col_profile.dtype}")
    print(f"  Null %: {col_profile.null_percentage:.1f}%")
    print(f"  Unique: {col_profile.unique_count}")
```

## ML Pipeline

### Feature Engineering

```python
from dataforge import FeatureEngineer

engineer = FeatureEngineer()

# Scale numeric features
scaled_df, scalers = engineer.scale_features(
    df,
    columns=["age", "income"],
    method="standard"  # or "minmax", "robust"
)

# Encode categorical features
encoded_df, encoders = engineer.encode_categorical(
    df,
    columns=["category", "region"],
    method="onehot"  # or "label"
)
```

### MLflow Integration

```python
from dataforge import MLflowTracker

tracker = MLflowTracker(
    experiment_name="my_experiment",
    tracking_uri="databricks"  # or local URI
)

with tracker.start_run("training_run"):
    # Log parameters
    tracker.log_params({"n_estimators": 100, "max_depth": 10})

    # Train model...

    # Log metrics
    tracker.log_metrics({"accuracy": 0.95, "f1": 0.92})

    # Log model
    tracker.log_model(model, "random_forest")
```

## Streaming

### Structured Streaming

```python
from dataforge.streaming.sources import KafkaSource
from dataforge.streaming.sinks import DeltaSink
from dataforge.streaming.processors import StreamProcessor

# Configure source
source = KafkaSource(
    bootstrap_servers="broker:9092",
    topic="events",
    starting_offsets="latest"
)

# Configure sink
sink = DeltaSink(
    table_name="catalog.schema.events",
    checkpoint_location="/checkpoints/events"
)

# Process stream
processor = StreamProcessor(
    watermark_column="event_time",
    watermark_delay="10 minutes"
)
```

### Windowed Aggregations

```python
from pyspark.sql.functions import window

# Tumbling window
df.groupBy(window("event_time", "5 minutes")).count()

# Sliding window
df.groupBy(window("event_time", "10 minutes", "5 minutes")).count()

# Session window
df.groupBy(session_window("event_time", "5 minutes")).count()
```

## Databricks Integration

### Delta Lake

```python
from dataforge.databricks import DeltaTableManager

delta = DeltaTableManager(spark)

# Optimize table with Z-ordering
delta.optimize(
    table_name="catalog.schema.orders",
    z_order_columns=["customer_id", "order_date"]
)

# Vacuum old files
delta.vacuum("catalog.schema.orders", retention_hours=168)

# Time travel query
historical_df = delta.read_version("catalog.schema.orders", version=10)

# Enable Change Data Feed
delta.enable_change_data_feed("catalog.schema.orders")

# Read changes
changes = delta.read_changes(
    table_name="catalog.schema.orders",
    starting_version=100
)
```

### MERGE Operations

```python
# Upsert with merge
delta.merge(
    target_table="catalog.schema.customers",
    source_df=updates_df,
    merge_keys=["customer_id"],
    update_columns={"name": "source.name", "email": "source.email"},
    insert_all=True
)

# SCD Type 2 merge
delta.merge_scd2(
    target_table="catalog.schema.dim_customers",
    source_df=updates_df,
    key_columns=["customer_id"],
    track_columns=["name", "address", "status"]
)
```

### Unity Catalog

Unity Catalog provides centralized governance for all data assets in Databricks.

#### Three-Level Namespace

```
catalog.schema.table
   │       │      │
   │       │      └── Table/View name
   │       └── Schema (database) name
   └── Catalog name
```

#### Catalog Management

```python
from dataforge.databricks import UnityCatalogManager

uc = UnityCatalogManager(spark)

# Create catalog
uc.create_catalog("prod", comment="Production data catalog")

# Create schema
uc.create_schema("prod", "sales", comment="Sales department data")

# Create managed table
uc.create_managed_table(
    full_table_name="prod.sales.orders",
    df=orders_df,
    comment="Customer orders table",
    partition_columns=["order_date"]
)
```

#### Permission Management

```python
# Grant SELECT on table
uc.grant_select("prod.sales.orders", "data_analysts")

# Grant multiple permissions
uc.grant_permissions(
    "prod.sales.orders",
    "data_engineers",
    ["SELECT", "MODIFY"]
)

# Grant schema-level permissions
uc.grant_schema_usage("prod", "sales", "data_analysts")

# Show grants
grants = uc.show_grants("TABLE", "prod.sales.orders")
```

#### Data Lineage

```python
# Get table lineage
lineage = uc.get_table_lineage("prod.sales.orders")

# Get column lineage
column_lineage = uc.get_column_lineage(
    "prod.sales.orders",
    "total_amount"
)
```

#### External Locations and Volumes

```python
# Create external location
uc.create_external_location(
    name="s3_landing",
    url="s3://bucket/landing/",
    credential_name="aws_credential"
)

# Create volume for file storage
uc.create_volume(
    catalog="prod",
    schema="raw_data",
    volume_name="landing_zone",
    location="s3://bucket/landing/"
)
```

#### Best Practices for Unity Catalog

1. **Naming Conventions**
   - Use lowercase with underscores
   - Catalogs: `prod`, `dev`, `staging`
   - Schemas: Business domain names (`sales`, `marketing`, `finance`)
   - Tables: Descriptive names (`customer_orders`, `daily_revenue`)

2. **Security**
   - Grant minimum required permissions
   - Use groups instead of individual users
   - Implement row-level security for sensitive data
   - Enable audit logging

3. **Organization**
   ```
   prod/
   ├── sales/
   │   ├── orders
   │   ├── customers
   │   └── products
   ├── marketing/
   │   ├── campaigns
   │   └── events
   └── finance/
       ├── transactions
       └── reports
   ```

4. **Documentation**
   - Add comments to all objects
   - Document column definitions
   - Maintain data dictionary

### Photon

```python
from dataforge.databricks import PhotonAnalyzer

analyzer = PhotonAnalyzer(spark)

# Check Photon compatibility
compatibility = analyzer.analyze_query(df)
print(f"Photon compatible: {compatibility.is_compatible}")
print(f"Recommendations: {compatibility.recommendations}")

# Get optimization suggestions
suggestions = analyzer.get_optimization_suggestions()
```

## S3 Storage Optimization

DataForge provides comprehensive S3 storage analysis and optimization capabilities.

### S3 Optimizer

```python
from dataforge.storage import S3Optimizer

optimizer = S3Optimizer(spark)

# Apply optimal S3 configuration
optimizer.apply_optimal_config(workload_type="analytics")

# Analyze S3 path for performance issues
report = optimizer.analyze_path("s3://my-bucket/data/")
print(f"Performance Score: {report.overall_score}/100")
print(f"Total Files: {report.total_files}")
print(f"Avg File Size: {report.avg_file_size_mb:.1f} MB")

# View issues and recommendations
for issue in report.issues:
    print(f"[{issue.severity}] {issue.description}")
    print(f"  Recommendation: {issue.recommendation}")

for rec in report.recommendations:
    print(f"  - {rec}")
```

### File Compaction

```python
# Compact small files for better performance
result = optimizer.compact_files(
    source_path="s3://bucket/raw-data/",
    target_path="s3://bucket/compacted-data/",
    target_file_size_mb=256,
    file_format="parquet",
    compression="snappy"
)
print(f"Reduced from {result['input_files']} to {result['output_files']} files")
```

### Format Conversion

```python
# Convert CSV/JSON to Parquet for better performance
result = optimizer.convert_format(
    source_path="s3://bucket/csv-data/",
    target_path="s3://bucket/parquet-data/",
    source_format="csv",
    target_format="parquet",
    compression="snappy",
    partition_by=["date", "region"]
)
```

### Storage Cost Estimation

```python
# Estimate savings from format conversion
savings = optimizer.estimate_cost_savings(
    current_size_gb=1000,
    current_format="csv",
    target_format="parquet"
)
print(f"Estimated new size: {savings['estimated_new_size_gb']:.1f} GB")
print(f"Monthly savings: ${savings['monthly_savings_usd']:.2f}")
print(f"Yearly savings: ${savings['yearly_savings_usd']:.2f}")
```

### S3 Configuration Best Practices

```python
# Optimal S3A settings applied by S3Optimizer:
{
    # Connection pooling
    "spark.hadoop.fs.s3a.connection.maximum": "100",

    # Fast upload for better write performance
    "spark.hadoop.fs.s3a.fast.upload": "true",
    "spark.hadoop.fs.s3a.fast.upload.buffer": "disk",

    # Magic committer for consistent writes
    "spark.hadoop.fs.s3a.committer.name": "magic",
    "spark.hadoop.fs.s3a.committer.magic.enabled": "true",

    # Multipart upload optimization
    "spark.hadoop.fs.s3a.multipart.threshold": "67108864",  # 64MB
    "spark.hadoop.fs.s3a.multipart.size": "67108864",

    # Block size for optimal reads
    "spark.hadoop.fs.s3a.block.size": "128M",
}
```

### S3 Performance Guidelines

| Issue | Impact | Solution |
|-------|--------|----------|
| Small files (< 32MB) | Slow reads, high API costs | Compact to 128MB-1GB files |
| CSV/JSON format | Poor compression, no column pruning | Convert to Parquet/Delta |
| Too many partitions | Slow listing, many small files | Reduce partition granularity |
| No compression | High storage costs | Use Snappy or ZSTD |
| Wrong storage class | Unnecessary costs | Use Intelligent-Tiering |

### Storage Class Recommendations

```python
# Get storage class recommendation
storage_class = optimizer.get_storage_class_recommendation(
    access_frequency="infrequent",  # frequent, infrequent, rare, archive
    data_criticality="important"     # critical, important, low
)
print(f"Recommended: {storage_class.value}")
# Output: STANDARD_IA
```

| Access Pattern | Criticality | Recommended Class |
|----------------|-------------|-------------------|
| Frequent | Any | STANDARD |
| Infrequent | Critical | STANDARD_IA |
| Infrequent | Low | ONEZONE_IA |
| Rare | Any | GLACIER_INSTANT_RETRIEVAL |
| Archive | Critical | GLACIER |
| Archive | Low | DEEP_ARCHIVE |

### Format Advisor

```python
from dataforge.storage import FormatAdvisor, UseCase

advisor = FormatAdvisor()

# Get format recommendation
rec = advisor.recommend(
    use_case=UseCase.ANALYTICS,
    data_size_gb=100,
    need_acid=True,
    need_time_travel=True
)

print(f"Recommended: {rec.format.value}")
print(f"Score: {rec.score}/100")
print(f"Compression: {rec.compression}")

for reason in rec.reasons:
    print(f"  - {reason}")
```

## Scala Library

DataForge includes a Scala library for Databricks development:

### Delta Lake Utilities

```scala
import com.dataforge.DeltaLakeUtils

val deltaUtils = new DeltaLakeUtils(spark)

// Optimize with Z-ordering
deltaUtils.optimize(
  tableName = "prod.sales.orders",
  zOrderBy = Seq("customer_id", "order_date")
)

// Vacuum old files
deltaUtils.vacuum("prod.sales.orders", retentionHours = 168)

// Time travel
val historicalDf = deltaUtils.readVersion("prod.sales.orders", version = 10)

// Merge (upsert)
deltaUtils.merge(
  targetTable = "prod.sales.customers",
  sourceDF = updatesDF,
  mergeCondition = "target.customer_id = source.customer_id"
)
```

### Spark Optimizations

```scala
import com.dataforge.SparkOptimizer

val optimizer = new SparkOptimizer(spark)

// Apply recommended configuration
optimizer.applyRecommendedConfig()

// Configure for specific workload
optimizer.configureForETL(dataSize = "large")
optimizer.configureForStreaming(
  checkpointLocation = "/checkpoints",
  triggerInterval = "1 minute"
)

// Optimize joins
val result = optimizer.optimizeJoin(leftDf, rightDf, Seq("key"), "inner")
```

### Unity Catalog Utilities

```scala
import com.dataforge.UnityCatalogManager

val ucManager = new UnityCatalogManager(spark)

// Create catalog and schema
ucManager.createCatalog("prod", Some("Production catalog"))
ucManager.createSchema("prod", "sales", Some("Sales data"))

// Create managed table
ucManager.createManagedTable(
  fullTableName = "prod.sales.orders",
  df = ordersDF,
  comment = Some("Customer orders"),
  partitionBy = Some(Seq("order_date"))
)

// Grant permissions
ucManager.grantSelect("prod.sales.orders", "data_analysts")
```

### Streaming Utilities

```scala
import com.dataforge.StreamingUtils

val streaming = new StreamingUtils(spark)

// Read from Kafka
val events = streaming.readFromKafka(
  bootstrapServers = "broker:9092",
  topic = "events",
  schema = eventSchema
)

// Window aggregation
val aggregated = streaming.windowAggregate(
  df = events,
  timestampCol = "event_time",
  windowSize = "5 minutes",
  aggregations = Map("value" -> "sum", "id" -> "count"),
  watermark = Some("10 minutes")
)

// Write to Delta
streaming.writeToDelta(
  df = aggregated,
  tableName = "prod.events.metrics",
  checkpointLocation = "/checkpoints/events",
  triggerInterval = "1 minute"
)
```

### Building the Scala Library

```bash
cd scala
sbt compile
sbt assembly  # Creates fat JAR
```

## Best Practices

### Pandas Optimizations

```python
# 1. Use appropriate dtypes
df['category'] = df['category'].astype('category')
df['count'] = df['count'].astype('int32')  # Instead of int64

# 2. Use query() for filtering (leverages numexpr)
df.query('age > 30 and salary > 50000')

# 3. Avoid iterrows() - use vectorized operations
df['bonus'] = df['salary'] * 0.1  # Good
# for idx, row in df.iterrows(): ...  # Bad

# 4. Read large files in chunks
for chunk in pd.read_csv('large.csv', chunksize=100000):
    process(chunk)

# 5. Enable copy-on-write (pandas 2.0+)
pd.options.mode.copy_on_write = True
```

### Polars Optimizations

```python
# 1. Use lazy evaluation for query optimization
import polars as pl
lf = pl.scan_parquet("data.parquet")
result = (
    lf.filter(pl.col("age") > 30)
    .group_by("city")
    .agg(pl.col("salary").mean())
    .collect()  # Execute optimized plan
)

# 2. Use streaming for large datasets
result = (
    pl.scan_parquet("large_data.parquet")
    .filter(pl.col("status") == "active")
    .collect(streaming=True)  # Process in chunks
)

# 3. Use expressions instead of apply()
df.with_columns(
    pl.col("name").str.to_uppercase().alias("name_upper"),
    (pl.col("price") * pl.col("quantity")).alias("total"),
)

# 4. Leverage multi-threaded execution (automatic)
# Polars uses all CPU cores by default

# 5. Use sink_parquet for memory-efficient writes
lf.sink_parquet("output.parquet")
```

### Spark Optimizations

```python
# 1. Enable Adaptive Query Execution
spark.conf.set("spark.sql.adaptive.enabled", "true")

# 2. Use broadcast for small tables
from pyspark.sql.functions import broadcast
df.join(broadcast(small_df), "key")

# 3. Avoid UDFs - use native functions
from pyspark.sql.functions import col, when
df.withColumn("status", when(col("age") > 30, "senior").otherwise("junior"))

# 4. Cache strategically and unpersist
df.cache()
df.count()  # Trigger caching
# ... use df multiple times ...
df.unpersist()  # Release memory

# 5. Partition appropriately (~128MB per partition)
df.repartition(200)  # Based on data size
```

### RAPIDS Optimizations

```python
# 1. Minimize CPU-GPU transfers
# Keep data on GPU throughout processing

# 2. Use cuDF I/O for better performance
import cudf
df = cudf.read_parquet('data.parquet')  # Faster than pandas

# 3. Configure memory pool
import rmm
rmm.reinitialize(pool_allocator=True, initial_pool_size=2**30)

# 4. Handle GPU memory limits
try:
    result = process_on_gpu(df)
except MemoryError:
    result = process_on_cpu(df.to_pandas())
```

### Delta Lake Optimizations

```python
# 1. OPTIMIZE regularly for write-heavy tables
spark.sql("OPTIMIZE catalog.schema.table")

# 2. Use Z-ORDER on frequently filtered columns
spark.sql("OPTIMIZE catalog.schema.table ZORDER BY (date, region)")

# 3. VACUUM old files (minimum 7 days retention)
spark.sql("VACUUM catalog.schema.table RETAIN 168 HOURS")

# 4. Enable auto-optimization
spark.sql("""
    ALTER TABLE catalog.schema.table
    SET TBLPROPERTIES (
        'delta.autoOptimize.optimizeWrite' = 'true',
        'delta.autoOptimize.autoCompact' = 'true'
    )
""")

# 5. Use Change Data Feed for incremental processing
spark.sql("""
    ALTER TABLE catalog.schema.table
    SET TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true')
""")
```

## API Reference

### Core Classes

| Class | Description |
|-------|-------------|
| `DataFrame` | Unified DataFrame interface |
| `EngineRecommender` | Engine selection advisor |
| `DataForgeConfig` | Configuration management |

### Quality Classes

| Class | Description |
|-------|-------------|
| `SchemaValidator` | Schema validation |
| `ColumnValidator` | Column-level validation |
| `DataProfiler` | Data profiling |

### Databricks Classes

| Class | Description |
|-------|-------------|
| `DeltaTableManager` | Delta Lake operations |
| `UnityCatalogManager` | Unity Catalog management |
| `PhotonAnalyzer` | Photon compatibility analysis |

### ML Classes

| Class | Description |
|-------|-------------|
| `FeatureEngineer` | Feature engineering utilities |
| `MLPipeline` | ML pipeline construction |
| `MLflowTracker` | MLflow integration |

## Examples

See the `examples/` directory for complete examples:

- `01_basic_usage.py` - Basic DataFrame operations
- `02_engine_comparison.py` - Engine benchmarking
- `03_data_quality.py` - Data validation and profiling
- `04_ml_pipeline.py` - ML feature engineering
- `05_streaming.py` - Structured Streaming patterns

## Code Validation

### Type Checking with ty

This project uses [ty](https://github.com/astral-sh/ty) — an extremely fast Python type checker from Astral (makers of Ruff), written in Rust:

```bash
# Install ty
pip install ty
# or with uv
uv tool install ty

# Run type checking
ty check dataforge/

# Check specific module
ty check dataforge/engines/polars_engine.py
```

### Linting with Ruff

```bash
# Install ruff
pip install ruff

# Lint
ruff check dataforge/

# Auto-fix
ruff check dataforge/ --fix
```

### Pre-commit Hooks

```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install

# Run against all files
pre-commit run --all-files
```

### Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=dataforge --cov-report=html

# Run specific test
python -m pytest tests/test_core.py -v
```

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Apache Spark team for PySpark
- Polars team for the high-performance DataFrame library
- NVIDIA RAPIDS team for cuDF
- Databricks team for Delta Lake and Unity Catalog
- pandas development team

---

**DataForge** - Making data processing intelligent and effortless.
