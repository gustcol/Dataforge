/**
 * DataForge Delta Lake Utilities for Scala
 *
 * Best practices for Delta Lake operations in Databricks.
 *
 * Key Features:
 *   - OPTIMIZE with Z-ORDER
 *   - VACUUM management
 *   - Time travel queries
 *   - Change Data Feed
 *   - Merge (UPSERT) operations
 *
 * Best Practices:
 *   1. Run OPTIMIZE regularly (daily for write-heavy tables)
 *   2. Use Z-ORDER on frequently filtered columns
 *   3. Set appropriate VACUUM retention (min 7 days)
 *   4. Enable Change Data Feed for incremental processing
 *   5. Use merge for efficient upserts
 *
 * @author DataForge Team
 */

package com.dataforge

import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import io.delta.tables.DeltaTable

/**
 * Delta Lake table manager for optimization and maintenance operations.
 *
 * Example usage:
 * {{{
 * val deltaUtils = new DeltaLakeUtils(spark)
 *
 * // Optimize table with Z-ordering
 * deltaUtils.optimize("catalog.schema.table", Seq("date", "region"))
 *
 * // Vacuum old files
 * deltaUtils.vacuum("catalog.schema.table", retentionHours = 168)
 *
 * // Time travel query
 * val historicalDf = deltaUtils.readVersion("catalog.schema.table", version = 10)
 * }}}
 */
class DeltaLakeUtils(spark: SparkSession) {

  import spark.implicits._

  // =========================================================================
  // OPTIMIZATION OPERATIONS
  // =========================================================================

  /**
   * Optimize Delta table by compacting small files.
   *
   * OPTIMIZE consolidates many small files into fewer larger files,
   * significantly improving read performance.
   *
   * @param tableName Full table name (catalog.schema.table)
   * @param zOrderBy  Optional columns for Z-ordering (improves query performance)
   * @param where     Optional partition filter
   * @return Optimization metrics
   *
   * Best Practice: Run during off-peak hours for large tables.
   *
   * Example:
   * {{{
   * // Basic optimize
   * deltaUtils.optimize("prod.sales.orders")
   *
   * // With Z-ordering for better query performance
   * deltaUtils.optimize("prod.sales.orders", Seq("customer_id", "order_date"))
   *
   * // Optimize specific partition
   * deltaUtils.optimize("prod.sales.orders", where = Some("order_date >= '2024-01-01'"))
   * }}}
   */
  def optimize(
      tableName: String,
      zOrderBy: Seq[String] = Seq.empty,
      where: Option[String] = None
  ): DataFrame = {

    var sql = s"OPTIMIZE $tableName"

    where.foreach(w => sql += s" WHERE $w")

    if (zOrderBy.nonEmpty) {
      sql += s" ZORDER BY (${zOrderBy.mkString(", ")})"
    }

    println(s"Running: $sql")
    spark.sql(sql)
  }

  /**
   * Remove old files from Delta table.
   *
   * VACUUM removes files no longer referenced by the Delta table
   * that are older than the retention period.
   *
   * @param tableName      Full table name
   * @param retentionHours Minimum file age to delete (default: 168 hours / 7 days)
   *
   * Warning:
   *   - Never set retention < 168 hours in production
   *   - Time travel won't work for removed versions
   *   - Concurrent readers may fail if retention too low
   *
   * Example:
   * {{{
   * // Standard 7-day retention
   * deltaUtils.vacuum("prod.sales.orders")
   *
   * // 30-day retention for compliance
   * deltaUtils.vacuum("prod.sales.orders", retentionHours = 720)
   * }}}
   */
  def vacuum(tableName: String, retentionHours: Int = 168): Unit = {
    if (retentionHours < 168) {
      println(s"WARNING: Retention ${retentionHours}h is below recommended 168h (7 days)")
    }

    val sql = s"VACUUM $tableName RETAIN $retentionHours HOURS"
    println(s"Running: $sql")
    spark.sql(sql)
  }

  /**
   * Enable auto-compaction for a table.
   *
   * Auto-compaction automatically runs OPTIMIZE after writes,
   * preventing small file accumulation.
   *
   * Best Practice: Enable for streaming tables.
   */
  def enableAutoCompact(tableName: String): Unit = {
    spark.sql(s"""
      ALTER TABLE $tableName
      SET TBLPROPERTIES ('delta.autoOptimize.autoCompact' = 'true')
    """)
    println(s"Auto-compact enabled for $tableName")
  }

  /**
   * Enable optimized writes for a table.
   *
   * Optimized writes reduce the number of files created during writes.
   */
  def enableOptimizeWrite(tableName: String): Unit = {
    spark.sql(s"""
      ALTER TABLE $tableName
      SET TBLPROPERTIES ('delta.autoOptimize.optimizeWrite' = 'true')
    """)
    println(s"Optimize write enabled for $tableName")
  }

  // =========================================================================
  // TIME TRAVEL OPERATIONS
  // =========================================================================

  /**
   * Read a specific version of a Delta table.
   *
   * @param tableName Full table name
   * @param version   Version number to read
   * @return DataFrame at specified version
   *
   * Example:
   * {{{
   * val df = deltaUtils.readVersion("prod.sales.orders", version = 10)
   * }}}
   */
  def readVersion(tableName: String, version: Long): DataFrame = {
    spark.read
      .format("delta")
      .option("versionAsOf", version)
      .table(tableName)
  }

  /**
   * Read Delta table as of a specific timestamp.
   *
   * @param tableName Full table name
   * @param timestamp Timestamp string (e.g., "2024-01-01 00:00:00")
   * @return DataFrame at specified timestamp
   */
  def readTimestamp(tableName: String, timestamp: String): DataFrame = {
    spark.read
      .format("delta")
      .option("timestampAsOf", timestamp)
      .table(tableName)
  }

  /**
   * Get version history of a Delta table.
   *
   * @param tableName Full table name
   * @param limit     Maximum entries to return
   * @return DataFrame with history
   */
  def getHistory(tableName: String, limit: Option[Int] = None): DataFrame = {
    var sql = s"DESCRIBE HISTORY $tableName"
    limit.foreach(l => sql += s" LIMIT $l")
    spark.sql(sql)
  }

  /**
   * Restore Delta table to a previous version.
   *
   * @param tableName Full table name
   * @param version   Version to restore to (use None for timestamp)
   * @param timestamp Timestamp to restore to (use None for version)
   */
  def restore(
      tableName: String,
      version: Option[Long] = None,
      timestamp: Option[String] = None
  ): Unit = {
    val sql = (version, timestamp) match {
      case (Some(v), _) => s"RESTORE TABLE $tableName TO VERSION AS OF $v"
      case (_, Some(t)) => s"RESTORE TABLE $tableName TO TIMESTAMP AS OF '$t'"
      case _            => throw new IllegalArgumentException("Either version or timestamp required")
    }

    println(s"Running: $sql")
    spark.sql(sql)
  }

  // =========================================================================
  // MERGE OPERATIONS
  // =========================================================================

  /**
   * Perform MERGE (upsert) operation.
   *
   * MERGE allows atomic INSERT, UPDATE, and DELETE in a single operation.
   *
   * @param targetTable    Target Delta table name
   * @param sourceDF       Source DataFrame
   * @param mergeCondition Join condition (e.g., "target.id = source.id")
   * @param updateColumns  Columns to update on match (None = update all)
   * @param insertColumns  Columns to insert when not matched (None = insert all)
   *
   * Example:
   * {{{
   * deltaUtils.merge(
   *   targetTable = "prod.sales.customers",
   *   sourceDF = updatesDF,
   *   mergeCondition = "target.customer_id = source.customer_id",
   *   updateColumns = Some(Map(
   *     "name" -> "source.name",
   *     "email" -> "source.email",
   *     "updated_at" -> "current_timestamp()"
   *   ))
   * )
   * }}}
   */
  def merge(
      targetTable: String,
      sourceDF: DataFrame,
      mergeCondition: String,
      updateColumns: Option[Map[String, String]] = None,
      insertColumns: Option[Map[String, String]] = None
  ): Unit = {

    val deltaTable = DeltaTable.forName(spark, targetTable)

    var mergeBuilder = deltaTable
      .as("target")
      .merge(sourceDF.as("source"), mergeCondition)

    // When matched - update
    mergeBuilder = updateColumns match {
      case Some(cols) => mergeBuilder.whenMatched.updateExpr(cols)
      case None       => mergeBuilder.whenMatched.updateAll()
    }

    // When not matched - insert
    mergeBuilder = insertColumns match {
      case Some(cols) => mergeBuilder.whenNotMatched.insertExpr(cols)
      case None       => mergeBuilder.whenNotMatched.insertAll()
    }

    println(s"Executing MERGE on $targetTable")
    mergeBuilder.execute()
  }

  /**
   * Perform SCD Type 2 merge for slowly changing dimensions.
   *
   * @param targetTable    Target dimension table
   * @param sourceDF       Source DataFrame with updates
   * @param keyColumns     Business key columns
   * @param trackColumns   Columns to track for changes
   */
  def mergeSCD2(
      targetTable: String,
      sourceDF: DataFrame,
      keyColumns: Seq[String],
      trackColumns: Seq[String]
  ): Unit = {

    val keyCondition = keyColumns.map(c => s"target.$c = source.$c").mkString(" AND ")
    val changeCondition = trackColumns.map(c => s"target.$c <> source.$c").mkString(" OR ")

    val deltaTable = DeltaTable.forName(spark, targetTable)

    deltaTable
      .as("target")
      .merge(sourceDF.as("source"), keyCondition)
      .whenMatched(changeCondition)
      .updateExpr(Map(
        "is_current" -> "false",
        "end_date" -> "current_timestamp()"
      ))
      .whenNotMatched
      .insertAll()
      .execute()

    println(s"SCD2 merge completed on $targetTable")
  }

  // =========================================================================
  // CHANGE DATA FEED
  // =========================================================================

  /**
   * Enable Change Data Feed for a table.
   *
   * CDF tracks row-level changes for incremental processing.
   */
  def enableChangeDataFeed(tableName: String): Unit = {
    spark.sql(s"""
      ALTER TABLE $tableName
      SET TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true')
    """)
    println(s"Change Data Feed enabled for $tableName")
  }

  /**
   * Read changes from Change Data Feed.
   *
   * @param tableName       Full table name
   * @param startingVersion Start version (inclusive)
   * @param endingVersion   End version (inclusive, optional)
   * @return DataFrame with changes including _change_type column
   *
   * Example:
   * {{{
   * val changes = deltaUtils.readChanges("prod.sales.orders", 100, Some(110))
   * changes.filter($"_change_type" === "update_postimage").show()
   * }}}
   */
  def readChanges(
      tableName: String,
      startingVersion: Long,
      endingVersion: Option[Long] = None
  ): DataFrame = {

    var reader = spark.read
      .format("delta")
      .option("readChangeFeed", "true")
      .option("startingVersion", startingVersion)

    endingVersion.foreach(v => reader = reader.option("endingVersion", v))

    reader.table(tableName)
  }

  // =========================================================================
  // TABLE INFORMATION
  // =========================================================================

  /**
   * Get detailed information about a Delta table.
   */
  def getTableInfo(tableName: String): DataFrame = {
    spark.sql(s"DESCRIBE DETAIL $tableName")
  }

  /**
   * Get file statistics and optimization recommendations.
   */
  def getFileStats(tableName: String): Map[String, Any] = {
    val detail = spark.sql(s"DESCRIBE DETAIL $tableName").collect()(0)

    val numFiles = detail.getAs[Long]("numFiles")
    val sizeBytes = detail.getAs[Long]("sizeInBytes")
    val avgFileSize = if (numFiles > 0) sizeBytes / numFiles else 0L

    Map(
      "numFiles" -> numFiles,
      "totalSizeGB" -> sizeBytes / (1024.0 * 1024.0 * 1024.0),
      "avgFileSizeMB" -> avgFileSize / (1024.0 * 1024.0),
      "needsOptimization" -> (avgFileSize < 64 * 1024 * 1024),
      "recommendation" -> (if (avgFileSize < 64 * 1024 * 1024) "Run OPTIMIZE" else "Table is well optimized")
    )
  }
}

/**
 * Companion object with utility methods.
 */
object DeltaLakeUtils {

  /**
   * Create a DeltaLakeUtils instance from the active SparkSession.
   */
  def apply(): DeltaLakeUtils = {
    new DeltaLakeUtils(SparkSession.active)
  }

  /**
   * Recommended Spark configurations for Delta Lake.
   */
  def recommendedConfigs: Map[String, String] = Map(
    "spark.sql.extensions" -> "io.delta.sql.DeltaSparkSessionExtension",
    "spark.sql.catalog.spark_catalog" -> "org.apache.spark.sql.delta.catalog.DeltaCatalog",
    "spark.databricks.delta.optimizeWrite.enabled" -> "true",
    "spark.databricks.delta.autoCompact.enabled" -> "true",
    "spark.sql.adaptive.enabled" -> "true"
  )
}
