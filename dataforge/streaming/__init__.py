"""
DataForge Streaming Module

Structured Streaming utilities for Spark streaming workloads.

Features:
    - Stream sources (Kafka, files, Delta)
    - Stream sinks (Kafka, files, Delta, console)
    - Stream processors (transformations, aggregations)
    - Watermarking and windowing
    - Stateful processing patterns

Best Practices:
    1. Use checkpointing for fault tolerance
    2. Configure watermarks for late data handling
    3. Monitor trigger intervals
    4. Use Delta Lake for exactly-once semantics
    5. Leverage Auto Loader for file sources

Example:
    >>> from dataforge.streaming import KafkaSource, DeltaSink, StreamProcessor
    >>>
    >>> # Create source
    >>> source = KafkaSource(
    ...     spark,
    ...     bootstrap_servers="kafka:9092",
    ...     topics=["events"]
    ... )
    >>>
    >>> # Process stream
    >>> processor = StreamProcessor(source.read_stream())
    >>> processed = processor.with_watermark("timestamp", "10 minutes")
    >>> processed = processor.window_aggregate(
    ...     "timestamp", "5 minutes",
    ...     {"count": "count", "value": "sum"}
    ... )
    >>>
    >>> # Write to sink
    >>> sink = DeltaSink(spark, "catalog.schema.events")
    >>> query = sink.write_stream(processed.stream, checkpoint_path="/checkpoints")
"""

from dataforge.streaming.sources import (
    StreamSource,
    KafkaSource,
    FileSource,
    DeltaSource,
    RateSource,
)
from dataforge.streaming.sinks import (
    StreamSink,
    KafkaSink,
    DeltaSink,
    FileSink,
    ConsoleSink,
    ForeachBatchSink,
)
from dataforge.streaming.processors import (
    StreamProcessor,
    WindowConfig,
    WatermarkConfig,
)

__all__ = [
    # Sources
    "StreamSource",
    "KafkaSource",
    "FileSource",
    "DeltaSource",
    "RateSource",
    # Sinks
    "StreamSink",
    "KafkaSink",
    "DeltaSink",
    "FileSink",
    "ConsoleSink",
    "ForeachBatchSink",
    # Processors
    "StreamProcessor",
    "WindowConfig",
    "WatermarkConfig",
]
