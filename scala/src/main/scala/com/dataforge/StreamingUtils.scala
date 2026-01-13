/**
 * DataForge Streaming Utilities for Scala
 *
 * Best practices for Structured Streaming in Databricks.
 *
 * Key Concepts:
 *   - Exactly-once processing semantics
 *   - Watermarking for late data handling
 *   - Window aggregations
 *   - Stateful processing
 *   - Multiple sink support
 *
 * Best Practices:
 *   1. Always set checkpoints for fault tolerance
 *   2. Use watermarks to handle late data
 *   3. Configure appropriate trigger intervals
 *   4. Monitor streaming metrics
 *   5. Use Delta Lake as sink for ACID guarantees
 *
 * @author DataForge Team
 */

package com.dataforge

import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.streaming.{StreamingQuery, Trigger, OutputMode}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.StructType
import scala.concurrent.duration._

/**
 * Streaming utilities for Databricks workloads.
 *
 * Example usage:
 * {{{
 * val streaming = new StreamingUtils(spark)
 *
 * // Read from Kafka
 * val kafkaStream = streaming.readFromKafka(
 *   "broker:9092",
 *   "events",
 *   schema
 * )
 *
 * // Process with windowing
 * val processed = streaming.windowAggregate(
 *   kafkaStream,
 *   "event_time",
 *   "5 minutes",
 *   Map("value" -> "count")
 * )
 *
 * // Write to Delta
 * streaming.writeToDelta(processed, "prod.events.aggregated", "1 minute")
 * }}}
 */
class StreamingUtils(spark: SparkSession) {

  import spark.implicits._

  // =========================================================================
  // KAFKA SOURCES
  // =========================================================================

  /**
   * Read from Kafka topic with JSON parsing.
   *
   * @param bootstrapServers Kafka bootstrap servers
   * @param topic            Kafka topic name
   * @param schema           Schema for JSON value parsing
   * @param startingOffsets  Starting offsets (earliest, latest, or specific)
   * @param options          Additional Kafka options
   * @return Streaming DataFrame with parsed data
   *
   * Example:
   * {{{
   * val schema = new StructType()
   *   .add("event_id", "string")
   *   .add("event_time", "timestamp")
   *   .add("user_id", "string")
   *   .add("action", "string")
   *
   * val stream = streaming.readFromKafka(
   *   "broker1:9092,broker2:9092",
   *   "user_events",
   *   schema
   * )
   * }}}
   */
  def readFromKafka(
      bootstrapServers: String,
      topic: String,
      schema: StructType,
      startingOffsets: String = "latest",
      options: Map[String, String] = Map.empty
  ): DataFrame = {

    var reader = spark.readStream
      .format("kafka")
      .option("kafka.bootstrap.servers", bootstrapServers)
      .option("subscribe", topic)
      .option("startingOffsets", startingOffsets)
      .option("failOnDataLoss", "false")

    options.foreach { case (k, v) => reader = reader.option(k, v) }

    val rawStream = reader.load()

    // Parse JSON value
    rawStream
      .selectExpr("CAST(key AS STRING)", "CAST(value AS STRING)", "timestamp", "topic", "partition", "offset")
      .select(
        col("key"),
        from_json(col("value"), schema).as("data"),
        col("timestamp").as("kafka_timestamp"),
        col("topic"),
        col("partition"),
        col("offset")
      )
      .select("key", "data.*", "kafka_timestamp", "topic", "partition", "offset")
  }

  /**
   * Read raw Kafka stream without parsing.
   *
   * @param bootstrapServers Kafka bootstrap servers
   * @param topic            Kafka topic name
   * @return Raw streaming DataFrame
   */
  def readFromKafkaRaw(
      bootstrapServers: String,
      topic: String
  ): DataFrame = {

    spark.readStream
      .format("kafka")
      .option("kafka.bootstrap.servers", bootstrapServers)
      .option("subscribe", topic)
      .option("startingOffsets", "latest")
      .load()
      .selectExpr(
        "CAST(key AS STRING) as key",
        "CAST(value AS STRING) as value",
        "timestamp",
        "topic",
        "partition",
        "offset"
      )
  }

  // =========================================================================
  // FILE SOURCES
  // =========================================================================

  /**
   * Read streaming from file source.
   *
   * @param path     Source directory path
   * @param format   File format (json, csv, parquet, delta)
   * @param schema   Schema for the data
   * @param options  Additional options (e.g., maxFilesPerTrigger)
   * @return Streaming DataFrame
   *
   * Example:
   * {{{
   * val stream = streaming.readFromFiles(
   *   "/data/landing/events/",
   *   "json",
   *   eventSchema,
   *   Map("maxFilesPerTrigger" -> "100")
   * )
   * }}}
   */
  def readFromFiles(
      path: String,
      format: String,
      schema: StructType,
      options: Map[String, String] = Map.empty
  ): DataFrame = {

    var reader = spark.readStream
      .format(format)
      .schema(schema)

    options.foreach { case (k, v) => reader = reader.option(k, v) }

    reader.load(path)
  }

  /**
   * Read Auto Loader stream (Databricks).
   *
   * Auto Loader efficiently processes new files as they arrive.
   *
   * @param path               Source path
   * @param format             Source file format
   * @param schemaLocation     Schema evolution location
   * @param checkpointLocation Checkpoint location
   * @return Streaming DataFrame
   *
   * Best Practice: Use Auto Loader for cloud storage ingestion.
   */
  def readWithAutoLoader(
      path: String,
      format: String,
      schemaLocation: String,
      checkpointLocation: String
  ): DataFrame = {

    spark.readStream
      .format("cloudFiles")
      .option("cloudFiles.format", format)
      .option("cloudFiles.schemaLocation", schemaLocation)
      .option("cloudFiles.schemaEvolutionMode", "addNewColumns")
      .load(path)
  }

  // =========================================================================
  // DELTA LAKE SOURCES
  // =========================================================================

  /**
   * Read streaming from Delta table.
   *
   * @param tableName Full table name
   * @param options   Additional options
   * @return Streaming DataFrame
   */
  def readFromDelta(
      tableName: String,
      options: Map[String, String] = Map.empty
  ): DataFrame = {

    var reader = spark.readStream
      .format("delta")

    options.foreach { case (k, v) => reader = reader.option(k, v) }

    reader.table(tableName)
  }

  /**
   * Read Change Data Feed from Delta table.
   *
   * @param tableName       Full table name
   * @param startingVersion Starting version for changes
   * @return Streaming DataFrame with changes
   *
   * Note: Table must have CDF enabled.
   */
  def readDeltaCDF(
      tableName: String,
      startingVersion: Long
  ): DataFrame = {

    spark.readStream
      .format("delta")
      .option("readChangeFeed", "true")
      .option("startingVersion", startingVersion)
      .table(tableName)
  }

  // =========================================================================
  // WINDOWING AND AGGREGATIONS
  // =========================================================================

  /**
   * Apply tumbling window aggregation.
   *
   * @param df            Streaming DataFrame
   * @param timestampCol  Timestamp column name
   * @param windowSize    Window size (e.g., "5 minutes")
   * @param aggregations  Map of column to aggregation function
   * @param watermark     Watermark duration for late data
   * @return Aggregated streaming DataFrame
   *
   * Example:
   * {{{
   * val aggregated = streaming.windowAggregate(
   *   eventsStream,
   *   "event_time",
   *   "5 minutes",
   *   Map("value" -> "sum", "count" -> "count"),
   *   Some("10 minutes")
   * )
   * }}}
   */
  def windowAggregate(
      df: DataFrame,
      timestampCol: String,
      windowSize: String,
      aggregations: Map[String, String],
      watermark: Option[String] = None
  ): DataFrame = {

    var stream = df
    watermark.foreach(w => stream = stream.withWatermark(timestampCol, w))

    val windowCol = window(col(timestampCol), windowSize)

    val aggExprs = aggregations.map { case (colName, aggFunc) =>
      aggFunc.toLowerCase match {
        case "count" => count(col(colName)).as(s"${colName}_count")
        case "sum"   => sum(col(colName)).as(s"${colName}_sum")
        case "avg"   => avg(col(colName)).as(s"${colName}_avg")
        case "min"   => min(col(colName)).as(s"${colName}_min")
        case "max"   => max(col(colName)).as(s"${colName}_max")
        case _       => count(col(colName)).as(s"${colName}_count")
      }
    }.toSeq

    stream
      .groupBy(windowCol)
      .agg(aggExprs.head, aggExprs.tail: _*)
      .select(
        col("window.start").as("window_start"),
        col("window.end").as("window_end"),
        col("*")
      )
      .drop("window")
  }

  /**
   * Apply sliding window aggregation.
   *
   * @param df           Streaming DataFrame
   * @param timestampCol Timestamp column name
   * @param windowSize   Window size
   * @param slideSize    Slide interval
   * @param aggregations Aggregation expressions
   * @param watermark    Watermark duration
   * @return Aggregated streaming DataFrame
   */
  def slidingWindowAggregate(
      df: DataFrame,
      timestampCol: String,
      windowSize: String,
      slideSize: String,
      aggregations: Map[String, String],
      watermark: Option[String] = None
  ): DataFrame = {

    var stream = df
    watermark.foreach(w => stream = stream.withWatermark(timestampCol, w))

    val windowCol = window(col(timestampCol), windowSize, slideSize)

    val aggExprs = aggregations.map { case (colName, aggFunc) =>
      aggFunc.toLowerCase match {
        case "count" => count(col(colName)).as(s"${colName}_count")
        case "sum"   => sum(col(colName)).as(s"${colName}_sum")
        case "avg"   => avg(col(colName)).as(s"${colName}_avg")
        case "min"   => min(col(colName)).as(s"${colName}_min")
        case "max"   => max(col(colName)).as(s"${colName}_max")
        case _       => count(col(colName)).as(s"${colName}_count")
      }
    }.toSeq

    stream
      .groupBy(windowCol)
      .agg(aggExprs.head, aggExprs.tail: _*)
  }

  /**
   * Apply session window aggregation.
   *
   * @param df           Streaming DataFrame
   * @param timestampCol Timestamp column
   * @param sessionKey   Session grouping column
   * @param gapDuration  Session gap duration
   * @param aggregations Aggregations to apply
   * @return Aggregated DataFrame
   */
  def sessionWindowAggregate(
      df: DataFrame,
      timestampCol: String,
      sessionKey: String,
      gapDuration: String,
      aggregations: Map[String, String]
  ): DataFrame = {

    val sessionWindow = session_window(col(timestampCol), gapDuration)

    val aggExprs = aggregations.map { case (colName, aggFunc) =>
      aggFunc.toLowerCase match {
        case "count" => count(col(colName)).as(s"${colName}_count")
        case "sum"   => sum(col(colName)).as(s"${colName}_sum")
        case _       => count(col(colName)).as(s"${colName}_count")
      }
    }.toSeq

    df.groupBy(col(sessionKey), sessionWindow)
      .agg(aggExprs.head, aggExprs.tail: _*)
  }

  // =========================================================================
  // SINKS
  // =========================================================================

  /**
   * Write stream to Delta Lake table.
   *
   * @param df                 Streaming DataFrame
   * @param tableName          Target table name
   * @param checkpointLocation Checkpoint location
   * @param triggerInterval    Trigger interval (e.g., "1 minute")
   * @param outputMode         Output mode (append, complete, update)
   * @return StreamingQuery
   *
   * Best Practice: Always use checkpoints for production streams.
   *
   * Example:
   * {{{
   * val query = streaming.writeToDelta(
   *   processedStream,
   *   "prod.events.processed",
   *   "/checkpoints/events",
   *   "1 minute"
   * )
   * }}}
   */
  def writeToDelta(
      df: DataFrame,
      tableName: String,
      checkpointLocation: String,
      triggerInterval: String = "1 minute",
      outputMode: String = "append"
  ): StreamingQuery = {

    val mode = outputMode.toLowerCase match {
      case "append"   => OutputMode.Append()
      case "complete" => OutputMode.Complete()
      case "update"   => OutputMode.Update()
      case _          => OutputMode.Append()
    }

    df.writeStream
      .format("delta")
      .outputMode(mode)
      .option("checkpointLocation", checkpointLocation)
      .trigger(Trigger.ProcessingTime(triggerInterval))
      .toTable(tableName)
  }

  /**
   * Write stream to Delta Lake with merge (upsert).
   *
   * @param df                 Streaming DataFrame
   * @param tableName          Target table name
   * @param mergeKeys          Keys for merge condition
   * @param checkpointLocation Checkpoint location
   * @return StreamingQuery
   */
  def writeToDeltaWithMerge(
      df: DataFrame,
      tableName: String,
      mergeKeys: Seq[String],
      checkpointLocation: String
  ): StreamingQuery = {

    import io.delta.tables.DeltaTable

    df.writeStream
      .format("delta")
      .foreachBatch { (batchDF: DataFrame, batchId: Long) =>
        val deltaTable = DeltaTable.forName(spark, tableName)
        val condition = mergeKeys.map(k => s"target.$k = source.$k").mkString(" AND ")

        deltaTable.as("target")
          .merge(batchDF.as("source"), condition)
          .whenMatched.updateAll()
          .whenNotMatched.insertAll()
          .execute()
      }
      .option("checkpointLocation", checkpointLocation)
      .start()
  }

  /**
   * Write stream to Kafka.
   *
   * @param df                 Streaming DataFrame with key and value columns
   * @param bootstrapServers   Kafka bootstrap servers
   * @param topic              Target topic
   * @param checkpointLocation Checkpoint location
   * @return StreamingQuery
   */
  def writeToKafka(
      df: DataFrame,
      bootstrapServers: String,
      topic: String,
      checkpointLocation: String
  ): StreamingQuery = {

    df.selectExpr("CAST(key AS STRING)", "to_json(struct(*)) AS value")
      .writeStream
      .format("kafka")
      .option("kafka.bootstrap.servers", bootstrapServers)
      .option("topic", topic)
      .option("checkpointLocation", checkpointLocation)
      .start()
  }

  /**
   * Write stream to console (for debugging).
   *
   * @param df              Streaming DataFrame
   * @param triggerInterval Trigger interval
   * @return StreamingQuery
   */
  def writeToConsole(
      df: DataFrame,
      triggerInterval: String = "10 seconds"
  ): StreamingQuery = {

    df.writeStream
      .format("console")
      .outputMode("append")
      .trigger(Trigger.ProcessingTime(triggerInterval))
      .start()
  }

  // =========================================================================
  // MONITORING
  // =========================================================================

  /**
   * Get active streaming queries.
   *
   * @return Array of active StreamingQuery
   */
  def getActiveQueries: Array[StreamingQuery] = {
    spark.streams.active
  }

  /**
   * Get streaming query progress.
   *
   * @param query StreamingQuery to monitor
   * @return Progress information as string
   */
  def getQueryProgress(query: StreamingQuery): String = {
    Option(query.lastProgress)
      .map(_.prettyJson)
      .getOrElse("No progress available")
  }

  /**
   * Stop all active streaming queries.
   */
  def stopAllQueries(): Unit = {
    spark.streams.active.foreach { query =>
      println(s"Stopping query: ${query.name}")
      query.stop()
    }
    println("All streaming queries stopped")
  }

  /**
   * Wait for query termination with timeout.
   *
   * @param query   StreamingQuery
   * @param timeout Timeout in milliseconds
   * @return true if query terminated, false if timeout
   */
  def awaitTermination(query: StreamingQuery, timeout: Long): Boolean = {
    query.awaitTermination(timeout)
  }

  // =========================================================================
  // UTILITY METHODS
  // =========================================================================

  /**
   * Add processing timestamp to stream.
   *
   * @param df Streaming DataFrame
   * @return DataFrame with processing_time column
   */
  def addProcessingTime(df: DataFrame): DataFrame = {
    df.withColumn("processing_time", current_timestamp())
  }

  /**
   * Deduplicate stream within a window.
   *
   * @param df           Streaming DataFrame
   * @param keys         Deduplication key columns
   * @param timestampCol Timestamp column for watermark
   * @param watermark    Watermark duration
   * @return Deduplicated stream
   */
  def deduplicateStream(
      df: DataFrame,
      keys: Seq[String],
      timestampCol: String,
      watermark: String
  ): DataFrame = {

    df.withWatermark(timestampCol, watermark)
      .dropDuplicates(keys)
  }

  /**
   * Rate limit stream processing.
   *
   * @param df               Streaming DataFrame
   * @param maxRecordsPerSec Maximum records per second
   * @return Rate-limited stream
   */
  def rateLimitStream(df: DataFrame, maxRecordsPerSec: Int): DataFrame = {
    // Note: This is a conceptual implementation
    // Actual rate limiting depends on source configuration
    df
  }
}

/**
 * Companion object with utility methods and constants.
 */
object StreamingUtils {

  /**
   * Create StreamingUtils from active SparkSession.
   */
  def apply(): StreamingUtils = {
    new StreamingUtils(SparkSession.active)
  }

  /**
   * Common trigger intervals.
   */
  object Triggers {
    val RealTime = "0 seconds"
    val TenSeconds = "10 seconds"
    val OneMinute = "1 minute"
    val FiveMinutes = "5 minutes"
    val TenMinutes = "10 minutes"
  }

  /**
   * Common watermark durations.
   */
  object Watermarks {
    val OneMinute = "1 minute"
    val FiveMinutes = "5 minutes"
    val TenMinutes = "10 minutes"
    val OneHour = "1 hour"
    val OneDay = "1 day"
  }

  /**
   * Output modes.
   */
  object OutputModes {
    val Append = "append"
    val Complete = "complete"
    val Update = "update"
  }
}
