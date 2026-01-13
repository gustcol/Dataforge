"""
DataForge Streaming Example

This example demonstrates:
    - Structured Streaming concepts
    - Stream sources and sinks
    - Windowing and aggregations
    - Watermarking for late data
    - Best practices for production streaming

Run this example:
    python examples/05_streaming.py

Note: Some features require PySpark with Structured Streaming support.

Author: DataForge Team
"""

import sys
sys.path.insert(0, '.')

from dataforge.streaming.sources import KafkaSource, FileSource, DeltaSource
from dataforge.streaming.sinks import DeltaSink, KafkaSink
from dataforge.streaming.processors import StreamProcessor
from dataforge.utils import setup_logging, get_logger

# Setup logging
setup_logging(level="INFO")
logger = get_logger(__name__)


def main():
    """Main entry point for streaming example."""

    print("=" * 70)
    print("DataForge Streaming Example")
    print("=" * 70)

    # =========================================================================
    # 1. Streaming Concepts
    # =========================================================================
    print("\n1. STRUCTURED STREAMING CONCEPTS")
    print("-" * 50)

    print("""
    Structured Streaming treats streaming data as an unbounded table:

    ┌─────────────────────────────────────────────────────────────────┐
    │                    STREAMING DATA MODEL                          │
    ├─────────────────────────────────────────────────────────────────┤
    │                                                                  │
    │   Input Stream (Unbounded Table)                                 │
    │   ┌─────────────────────────────────────────────────────────┐   │
    │   │ T1: { id: 1, value: 100, timestamp: "2024-01-01 10:00" }│   │
    │   │ T2: { id: 2, value: 200, timestamp: "2024-01-01 10:01" }│   │
    │   │ T3: { id: 3, value: 150, timestamp: "2024-01-01 10:02" }│   │
    │   │ ...                                                     │   │
    │   │ Tn: { new records keep arriving }                       │   │
    │   └─────────────────────────────────────────────────────────┘   │
    │                           │                                      │
    │                           ▼                                      │
    │                    ┌─────────────┐                               │
    │                    │   Query     │                               │
    │                    │ (Transform) │                               │
    │                    └─────────────┘                               │
    │                           │                                      │
    │                           ▼                                      │
    │   Output (Result Table)                                          │
    │   ┌─────────────────────────────────────────────────────────┐   │
    │   │ Continuously updated as new data arrives                │   │
    │   └─────────────────────────────────────────────────────────┘   │
    │                                                                  │
    └─────────────────────────────────────────────────────────────────┘

    Key Concepts:
    - Triggers: When to process new data (micro-batch or continuous)
    - Watermarks: How to handle late-arriving data
    - Output Modes: append, complete, update
    - Checkpoints: For fault tolerance and exactly-once semantics
    """)

    # =========================================================================
    # 2. Stream Sources
    # =========================================================================
    print("\n2. STREAM SOURCES")
    print("-" * 50)

    print("\n  Available Sources:")
    print("  " + "-" * 40)

    # Kafka Source
    kafka_source = KafkaSource(
        bootstrap_servers="localhost:9092",
        topic="events",
        starting_offsets="latest"
    )
    print(f"\n  Kafka Source:")
    print(f"    Servers: {kafka_source.bootstrap_servers}")
    print(f"    Topic: {kafka_source.topic}")
    print(f"    Offsets: {kafka_source.starting_offsets}")

    # File Source
    file_source = FileSource(
        path="/data/landing/events/",
        format="json",
        max_files_per_trigger=100
    )
    print(f"\n  File Source:")
    print(f"    Path: {file_source.path}")
    print(f"    Format: {file_source.format}")
    print(f"    Max files: {file_source.max_files_per_trigger}")

    # Delta Source
    delta_source = DeltaSource(
        table_name="catalog.schema.events",
        read_change_feed=True
    )
    print(f"\n  Delta Source:")
    print(f"    Table: {delta_source.table_name}")
    print(f"    CDF: {delta_source.read_change_feed}")

    # =========================================================================
    # 3. Stream Sinks
    # =========================================================================
    print("\n3. STREAM SINKS")
    print("-" * 50)

    print("\n  Available Sinks:")
    print("  " + "-" * 40)

    # Delta Sink
    delta_sink = DeltaSink(
        table_name="catalog.schema.processed_events",
        checkpoint_location="/checkpoints/events",
        output_mode="append"
    )
    print(f"\n  Delta Sink:")
    print(f"    Table: {delta_sink.table_name}")
    print(f"    Checkpoint: {delta_sink.checkpoint_location}")
    print(f"    Mode: {delta_sink.output_mode}")

    # Kafka Sink
    kafka_sink = KafkaSink(
        bootstrap_servers="localhost:9092",
        topic="processed_events",
        checkpoint_location="/checkpoints/kafka"
    )
    print(f"\n  Kafka Sink:")
    print(f"    Servers: {kafka_sink.bootstrap_servers}")
    print(f"    Topic: {kafka_sink.topic}")

    # =========================================================================
    # 4. Stream Processing
    # =========================================================================
    print("\n4. STREAM PROCESSING")
    print("-" * 50)

    processor = StreamProcessor()

    print("""
    Stream Processor Configuration:

    processor = StreamProcessor(
        watermark_column="event_time",
        watermark_delay="10 minutes",
        trigger_interval="1 minute"
    )

    Available Operations:
    - filter(): Filter records
    - select(): Project columns
    - with_column(): Add computed columns
    - groupby(): Aggregations
    - window(): Time-based windows
    - join(): Stream-stream or stream-static joins
    """)

    # =========================================================================
    # 5. Windowing
    # =========================================================================
    print("\n5. WINDOWING")
    print("-" * 50)

    print("""
    ┌────────────────────────────────────────────────────────────────────┐
    │                       WINDOW TYPES                                  │
    ├────────────────────────────────────────────────────────────────────┤
    │                                                                     │
    │  1. TUMBLING WINDOW (Non-overlapping, fixed-size)                  │
    │     ┌─────────┐┌─────────┐┌─────────┐┌─────────┐                   │
    │     │ 0-5 min ││ 5-10min ││10-15min ││15-20min │                   │
    │     └─────────┘└─────────┘└─────────┘└─────────┘                   │
    │                                                                     │
    │     Usage: Count events per 5-minute window                         │
    │     df.groupBy(window("timestamp", "5 minutes"))                    │
    │                                                                     │
    │  2. SLIDING WINDOW (Overlapping, fixed-size)                       │
    │     ┌─────────┐                                                    │
    │     │ 0-5 min │                                                    │
    │     └─────────┘                                                    │
    │        ┌─────────┐                                                 │
    │        │2-7 min  │                                                 │
    │        └─────────┘                                                 │
    │           ┌─────────┐                                              │
    │           │4-9 min  │                                              │
    │           └─────────┘                                              │
    │                                                                     │
    │     Usage: Moving average over 5 minutes, updated every 2 minutes   │
    │     df.groupBy(window("timestamp", "5 minutes", "2 minutes"))       │
    │                                                                     │
    │  3. SESSION WINDOW (Dynamic, gap-based)                            │
    │     ┌───────────────┐        ┌─────────┐     ┌───────────────┐    │
    │     │   Session 1   │ (gap)  │Session 2│(gap)│   Session 3   │    │
    │     └───────────────┘        └─────────┘     └───────────────┘    │
    │                                                                     │
    │     Usage: Group user activity with 5-minute gap                    │
    │     df.groupBy(session_window("timestamp", "5 minutes"))            │
    │                                                                     │
    └────────────────────────────────────────────────────────────────────┘
    """)

    # =========================================================================
    # 6. Watermarking
    # =========================================================================
    print("\n6. WATERMARKING (Late Data Handling)")
    print("-" * 50)

    print("""
    Watermarks define how late data can be:

    ┌────────────────────────────────────────────────────────────────────┐
    │                    WATERMARK CONCEPT                                │
    ├────────────────────────────────────────────────────────────────────┤
    │                                                                     │
    │   Event Time:    10:00  10:01  10:02  10:03  10:04  10:05          │
    │                    │      │      │      │      │      │            │
    │   Events:          E1     E2     E3    (E4)   E5     E6            │
    │                                   ▲      │                          │
    │                                   │      │                          │
    │                              Late Event (arrives at 10:05)         │
    │                                                                     │
    │   With Watermark = "2 minutes":                                    │
    │   - At 10:05, watermark = 10:03                                    │
    │   - E4 (10:03) is accepted (within watermark)                      │
    │   - Events before 10:03 would be dropped                           │
    │                                                                     │
    │   Code:                                                             │
    │   df.withWatermark("event_time", "2 minutes")                      │
    │     .groupBy(window("event_time", "5 minutes"))                    │
    │     .count()                                                        │
    │                                                                     │
    └────────────────────────────────────────────────────────────────────┘

    Best Practices:
    - Set watermark based on maximum expected delay
    - Balance between data completeness and memory usage
    - Monitor dropped late events
    """)

    # =========================================================================
    # 7. Output Modes
    # =========================================================================
    print("\n7. OUTPUT MODES")
    print("-" * 50)

    print("""
    ┌────────────────────────────────────────────────────────────────────┐
    │                      OUTPUT MODES                                   │
    ├────────────────────────────────────────────────────────────────────┤
    │                                                                     │
    │  1. APPEND MODE                                                     │
    │     - Only new rows added to result table                          │
    │     - Best for: Insert-only workloads                              │
    │     - Supports: Most queries without aggregation                    │
    │                                                                     │
    │     df.writeStream.outputMode("append")                            │
    │                                                                     │
    │  2. COMPLETE MODE                                                   │
    │     - Entire result table written each trigger                     │
    │     - Best for: Aggregations where results change                  │
    │     - Supports: All aggregation queries                            │
    │                                                                     │
    │     df.writeStream.outputMode("complete")                          │
    │                                                                     │
    │  3. UPDATE MODE                                                     │
    │     - Only changed rows written                                    │
    │     - Best for: Aggregations with watermark                        │
    │     - Supports: Most queries with state                            │
    │                                                                     │
    │     df.writeStream.outputMode("update")                            │
    │                                                                     │
    └────────────────────────────────────────────────────────────────────┘
    """)

    # =========================================================================
    # 8. Example Pipeline
    # =========================================================================
    print("\n8. EXAMPLE STREAMING PIPELINE")
    print("-" * 50)

    print("""
    # Complete streaming pipeline example:

    from pyspark.sql import SparkSession
    from pyspark.sql.functions import *

    spark = SparkSession.builder.getOrCreate()

    # 1. Read from Kafka
    events = (
        spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", "broker:9092")
        .option("subscribe", "user_events")
        .option("startingOffsets", "latest")
        .load()
        .selectExpr(
            "CAST(value AS STRING) as json_value",
            "timestamp as kafka_timestamp"
        )
        .select(
            from_json(col("json_value"), event_schema).alias("event"),
            col("kafka_timestamp")
        )
        .select("event.*", "kafka_timestamp")
    )

    # 2. Apply watermark and window aggregation
    aggregated = (
        events
        .withWatermark("event_time", "10 minutes")
        .groupBy(
            window("event_time", "5 minutes"),
            "event_type"
        )
        .agg(
            count("*").alias("event_count"),
            sum("value").alias("total_value"),
            approx_count_distinct("user_id").alias("unique_users")
        )
    )

    # 3. Write to Delta Lake
    query = (
        aggregated.writeStream
        .format("delta")
        .outputMode("append")
        .option("checkpointLocation", "/checkpoints/events")
        .trigger(processingTime="1 minute")
        .toTable("analytics.streaming.event_metrics")
    )

    # 4. Monitor the query
    query.awaitTermination()
    """)

    # =========================================================================
    # 9. Best Practices
    # =========================================================================
    print("\n9. STREAMING BEST PRACTICES")
    print("-" * 50)

    print("""
    ┌────────────────────────────────────────────────────────────────────┐
    │                PRODUCTION STREAMING BEST PRACTICES                  │
    ├────────────────────────────────────────────────────────────────────┤
    │                                                                     │
    │  1. CHECKPOINTING                                                   │
    │     - Always enable checkpoints for fault tolerance                │
    │     - Use reliable storage (S3, ADLS, DBFS)                        │
    │     - Never delete checkpoints of running queries                  │
    │                                                                     │
    │  2. WATERMARKING                                                    │
    │     - Set appropriate watermark for late data                      │
    │     - Monitor dropped late events                                  │
    │     - Balance completeness vs. memory                              │
    │                                                                     │
    │  3. TRIGGERS                                                        │
    │     - Use processingTime for batch-like behavior                   │
    │     - Use availableNow for one-time catch-up                       │
    │     - Avoid very short triggers (< 1 second)                       │
    │                                                                     │
    │  4. MONITORING                                                      │
    │     - Track inputRowsPerSecond                                     │
    │     - Monitor processedRowsPerSecond                               │
    │     - Alert on high watermark delay                                │
    │                                                                     │
    │  5. STATE MANAGEMENT                                                │
    │     - Use TTL for state to prevent unbounded growth                │
    │     - Choose appropriate state store (RocksDB for large state)     │
    │     - Monitor state size and cleanup                               │
    │                                                                     │
    │  6. ERROR HANDLING                                                  │
    │     - Use foreachBatch for custom error handling                   │
    │     - Implement dead-letter queues for bad records                 │
    │     - Set up alerting for query failures                           │
    │                                                                     │
    │  7. TESTING                                                         │
    │     - Unit test transformations with static data                   │
    │     - Integration test with rate source                            │
    │     - Load test before production deployment                       │
    │                                                                     │
    └────────────────────────────────────────────────────────────────────┘
    """)

    # =========================================================================
    # 10. Databricks-Specific Features
    # =========================================================================
    print("\n10. DATABRICKS STREAMING FEATURES")
    print("-" * 50)

    print("""
    Databricks provides additional streaming capabilities:

    1. AUTO LOADER
       - Efficient file ingestion
       - Automatic schema inference and evolution
       - Exactly-once guarantees

       spark.readStream.format("cloudFiles")
           .option("cloudFiles.format", "json")
           .option("cloudFiles.schemaLocation", "/schema")
           .load("/data/landing/")

    2. DELTA LIVE TABLES (DLT)
       - Declarative pipeline definition
       - Automatic dependency management
       - Built-in data quality expectations

       @dlt.table
       def silver_events():
           return dlt.read_stream("bronze_events")
               .filter("valid = true")

    3. PHOTON OPTIMIZATION
       - Native vectorized engine
       - Automatic optimization
       - 2-8x faster streaming

    4. STREAMING STATE MANAGEMENT
       - RocksDB-based state store
       - State snapshot backups
       - State rebalancing during scaling
    """)

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
    DataForge Streaming provides:

    1. SOURCES: Kafka, Files, Delta Lake, Rate (testing)
    2. SINKS: Delta Lake, Kafka, Files, Console
    3. PROCESSING: Filter, Select, Aggregate, Window, Join
    4. RELIABILITY: Checkpoints, Watermarks, Exactly-once
    5. MONITORING: Progress tracking, Metrics, Alerts

    Key Takeaways:
    - Always use checkpoints for production
    - Set watermarks for late data handling
    - Monitor query progress and metrics
    - Use Delta Lake for reliable streaming sinks
    - Test with rate source before production
    """)

    print("Streaming example completed!")


if __name__ == "__main__":
    main()
