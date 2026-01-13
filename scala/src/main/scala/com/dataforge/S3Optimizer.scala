/**
 * DataForge S3 Optimizer for Scala
 *
 * Best practices for S3 storage optimization in Databricks/Spark.
 *
 * Key Areas:
 *   - S3 configuration optimization
 *   - File size analysis and compaction
 *   - Format conversion recommendations
 *   - Storage class optimization
 *   - Cost reduction strategies
 *
 * Best Practices:
 *   1. Target file sizes of 128MB-1GB for optimal performance
 *   2. Use columnar formats (Parquet, Delta) for analytics
 *   3. Configure S3A connector for maximum throughput
 *   4. Implement appropriate partitioning strategies
 *   5. Use S3 Intelligent-Tiering for variable access patterns
 *
 * @author DataForge Team
 */

package com.dataforge

import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._

/**
 * S3 storage optimizer for Spark/Databricks workloads.
 *
 * Example usage:
 * {{{
 * val optimizer = new S3Optimizer(spark)
 *
 * // Apply optimal S3 configuration
 * optimizer.applyOptimalConfig()
 *
 * // Analyze S3 path
 * val report = optimizer.analyzePath("s3://bucket/data/")
 * println(s"Score: ${report.score}/100")
 *
 * // Compact small files
 * optimizer.compactFiles("s3://bucket/raw/", "s3://bucket/compacted/")
 * }}}
 */
class S3Optimizer(spark: SparkSession) {

  import spark.implicits._

  // Optimal file size range
  val OPTIMAL_MIN_SIZE_MB: Long = 128
  val OPTIMAL_MAX_SIZE_MB: Long = 1024
  val SMALL_FILE_THRESHOLD_MB: Long = 32

  // =========================================================================
  // S3 CONFIGURATION
  // =========================================================================

  /**
   * Apply optimal S3A configuration for Spark.
   *
   * Configures:
   *   - Connection pooling
   *   - Multipart uploads
   *   - Fast upload mode
   *   - Magic committer
   *   - Retry settings
   *
   * Best Practice: Call this at the start of your Spark application.
   */
  def applyOptimalConfig(): Map[String, String] = {
    val configs = Map(
      // Connection settings
      "spark.hadoop.fs.s3a.connection.maximum" -> "200",
      "spark.hadoop.fs.s3a.connection.timeout" -> "60000",
      "spark.hadoop.fs.s3a.socket.timeout" -> "60000",

      // Fast upload for better write performance
      "spark.hadoop.fs.s3a.fast.upload" -> "true",
      "spark.hadoop.fs.s3a.fast.upload.buffer" -> "disk",
      "spark.hadoop.fs.s3a.fast.upload.active.blocks" -> "8",

      // Multipart upload optimization
      "spark.hadoop.fs.s3a.multipart.threshold" -> "67108864", // 64MB
      "spark.hadoop.fs.s3a.multipart.size" -> "67108864",

      // Magic committer for consistent writes
      "spark.hadoop.fs.s3a.committer.name" -> "magic",
      "spark.hadoop.fs.s3a.committer.magic.enabled" -> "true",

      // Retry settings
      "spark.hadoop.fs.s3a.attempts.maximum" -> "10",
      "spark.hadoop.fs.s3a.retry.limit" -> "10",

      // Block size optimization
      "spark.hadoop.fs.s3a.block.size" -> "134217728", // 128MB

      // Read optimization
      "spark.hadoop.fs.s3a.readahead.range" -> "65536",
      "spark.hadoop.fs.s3a.experimental.input.fadvise" -> "random",

      // Parallelism
      "spark.hadoop.fs.s3a.threads.max" -> "64",
      "spark.hadoop.fs.s3a.max.total.tasks" -> "64"
    )

    configs.foreach { case (key, value) =>
      spark.conf.set(key, value)
    }

    println(s"Applied ${configs.size} S3 optimizations")
    configs
  }

  /**
   * Apply configuration optimized for ETL workloads.
   *
   * ETL workloads benefit from:
   *   - Sequential read optimization
   *   - Higher parallelism
   *   - Larger block sizes
   */
  def applyETLConfig(): Unit = {
    applyOptimalConfig()

    // ETL-specific overrides
    spark.conf.set("spark.hadoop.fs.s3a.experimental.input.fadvise", "sequential")
    spark.conf.set("spark.sql.files.maxPartitionBytes", "268435456") // 256MB
    spark.conf.set("spark.hadoop.fs.s3a.threads.max", "96")

    println("Applied ETL-optimized S3 configuration")
  }

  /**
   * Apply configuration optimized for analytics workloads.
   *
   * Analytics workloads benefit from:
   *   - Random access optimization
   *   - Predicate pushdown
   *   - Column pruning
   */
  def applyAnalyticsConfig(): Unit = {
    applyOptimalConfig()

    // Analytics-specific overrides
    spark.conf.set("spark.hadoop.fs.s3a.experimental.input.fadvise", "random")
    spark.conf.set("spark.sql.files.maxPartitionBytes", "134217728") // 128MB
    spark.conf.set("spark.sql.parquet.filterPushdown", "true")
    spark.conf.set("spark.sql.parquet.pushdown.inFilterThreshold", "10")

    println("Applied analytics-optimized S3 configuration")
  }

  /**
   * Get current S3-related configuration.
   *
   * @return Map of S3 configuration settings
   */
  def getCurrentConfig: Map[String, String] = {
    spark.conf.getAll
      .filter { case (k, _) => k.contains("s3") || k.contains("fs.s3a") }
      .toMap
  }

  /**
   * Print current S3 configuration.
   */
  def printConfig(): Unit = {
    println("\n=== S3 Configuration ===")
    getCurrentConfig.toSeq.sortBy(_._1).foreach { case (k, v) =>
      println(s"$k = $v")
    }
    println("========================\n")
  }

  // =========================================================================
  // PATH ANALYSIS
  // =========================================================================

  /**
   * Analyze S3 path for performance issues.
   *
   * @param path S3 path to analyze (s3://bucket/prefix/)
   * @return S3AnalysisReport with findings and recommendations
   *
   * Example:
   * {{{
   * val report = optimizer.analyzePath("s3://bucket/data/")
   * println(s"Total files: ${report.totalFiles}")
   * println(s"Avg size: ${report.avgSizeMB}MB")
   * report.issues.foreach(println)
   * }}}
   */
  def analyzePath(path: String): S3AnalysisReport = {
    try {
      val filesDF = spark.read.format("binaryFile").load(path)
        .select("path", "length")

      val stats = filesDF.agg(
        count("*").as("count"),
        sum("length").as("totalSize"),
        avg("length").as("avgSize"),
        min("length").as("minSize"),
        max("length").as("maxSize")
      ).collect()(0)

      val totalFiles = stats.getLong(0)
      val totalSizeBytes = stats.getLong(1)
      val avgSizeBytes = stats.getDouble(2)
      val minSizeBytes = stats.getLong(3)
      val maxSizeBytes = stats.getLong(4)

      // Convert to MB
      val avgSizeMB = avgSizeBytes / (1024.0 * 1024.0)
      val totalSizeGB = totalSizeBytes / (1024.0 * 1024.0 * 1024.0)

      // Count small files
      val smallFiles = filesDF
        .filter($"length" < SMALL_FILE_THRESHOLD_MB * 1024 * 1024)
        .count()

      // Detect file formats
      val formats = filesDF
        .withColumn("format",
          when($"path".endsWith(".parquet"), "parquet")
          .when($"path".endsWith(".orc"), "orc")
          .when($"path".contains(".json"), "json")
          .when($"path".contains(".csv"), "csv")
          .otherwise("other")
        )
        .groupBy("format")
        .count()
        .collect()
        .map(r => r.getString(0) -> r.getLong(1))
        .toMap

      // Generate issues
      val issues = scala.collection.mutable.ListBuffer[String]()

      if (smallFiles > totalFiles * 0.1) {
        issues += s"HIGH: ${smallFiles} files (${smallFiles * 100 / totalFiles}%) are under ${SMALL_FILE_THRESHOLD_MB}MB - consider compaction"
      }

      if (avgSizeMB < OPTIMAL_MIN_SIZE_MB) {
        issues += s"HIGH: Average file size (${avgSizeMB.toInt}MB) is below optimal (${OPTIMAL_MIN_SIZE_MB}MB)"
      }

      val rowFormats = formats.getOrElse("csv", 0L) + formats.getOrElse("json", 0L)
      if (rowFormats > totalFiles * 0.3) {
        issues += s"MEDIUM: ${rowFormats * 100 / totalFiles}% of files use row-based formats - consider converting to Parquet"
      }

      if (totalFiles > 10000) {
        issues += s"MEDIUM: High file count (${totalFiles}) may slow down file listing - consider reducing partitions"
      }

      // Calculate score
      var score = 100
      issues.foreach { issue =>
        if (issue.startsWith("HIGH")) score -= 20
        else if (issue.startsWith("MEDIUM")) score -= 10
        else score -= 5
      }

      // Bonus for optimal file size
      if (avgSizeMB >= OPTIMAL_MIN_SIZE_MB && avgSizeMB <= OPTIMAL_MAX_SIZE_MB) {
        score += 10
      }

      // Bonus for columnar formats
      val columnar = formats.getOrElse("parquet", 0L) + formats.getOrElse("orc", 0L)
      if (columnar > totalFiles * 0.8) {
        score += 10
      }

      S3AnalysisReport(
        path = path,
        totalFiles = totalFiles,
        totalSizeGB = totalSizeGB,
        avgSizeMB = avgSizeMB,
        minSizeMB = minSizeBytes / (1024.0 * 1024.0),
        maxSizeMB = maxSizeBytes / (1024.0 * 1024.0),
        smallFileCount = smallFiles,
        formats = formats,
        issues = issues.toList,
        score = Math.max(0, Math.min(100, score))
      )

    } catch {
      case e: Exception =>
        S3AnalysisReport(
          path = path,
          totalFiles = 0,
          totalSizeGB = 0,
          avgSizeMB = 0,
          minSizeMB = 0,
          maxSizeMB = 0,
          smallFileCount = 0,
          formats = Map.empty,
          issues = List(s"Error analyzing path: ${e.getMessage}"),
          score = 0
        )
    }
  }

  // =========================================================================
  // FILE COMPACTION
  // =========================================================================

  /**
   * Compact small files into optimally-sized files.
   *
   * @param sourcePath      Source S3 path with small files
   * @param targetPath      Target S3 path for compacted files
   * @param targetSizeMB    Target file size in MB (default: 256)
   * @param format          Output format (default: parquet)
   * @param compression     Compression codec (default: snappy)
   * @return Compaction result statistics
   *
   * Example:
   * {{{
   * val result = optimizer.compactFiles(
   *   "s3://bucket/raw/",
   *   "s3://bucket/compacted/",
   *   targetSizeMB = 256
   * )
   * println(s"Reduced from ${result.inputFiles} to ${result.outputFiles} files")
   * }}}
   */
  def compactFiles(
      sourcePath: String,
      targetPath: String,
      targetSizeMB: Int = 256,
      format: String = "parquet",
      compression: String = "snappy"
  ): CompactionResult = {

    // Read source data
    val df = spark.read.format(format).load(sourcePath)
    val inputFiles = df.inputFiles.length

    // Estimate total size and calculate optimal partitions
    val totalRows = df.count()
    val sampleSize = Math.min(1000, totalRows)
    val sampleSizeBytes = if (sampleSize > 0) {
      df.limit(sampleSize.toInt).rdd.map(_.toString.getBytes.length.toLong).sum()
    } else 0L

    val estimatedTotalBytes = if (sampleSize > 0) {
      (sampleSizeBytes.toDouble / sampleSize) * totalRows
    } else 0.0

    val targetBytes = targetSizeMB.toLong * 1024 * 1024
    val numPartitions = Math.max(1, (estimatedTotalBytes / targetBytes).toInt)

    // Compact and write
    val compacted = df.coalesce(numPartitions)

    compacted.write
      .format(format)
      .option("compression", compression)
      .mode("overwrite")
      .save(targetPath)

    println(s"Compacted $inputFiles files into $numPartitions files")

    CompactionResult(
      sourcePath = sourcePath,
      targetPath = targetPath,
      inputFiles = inputFiles,
      outputFiles = numPartitions,
      format = format,
      compression = compression
    )
  }

  /**
   * Compact files using Delta Lake OPTIMIZE.
   *
   * @param tableName Full table name (catalog.schema.table)
   * @param zOrderBy  Optional columns for Z-ordering
   *
   * Best Practice: Use Delta Lake OPTIMIZE for production tables.
   */
  def compactDeltaTable(
      tableName: String,
      zOrderBy: Seq[String] = Seq.empty
  ): Unit = {

    var sql = s"OPTIMIZE $tableName"

    if (zOrderBy.nonEmpty) {
      sql += s" ZORDER BY (${zOrderBy.mkString(", ")})"
    }

    println(s"Running: $sql")
    spark.sql(sql)
  }

  // =========================================================================
  // FORMAT CONVERSION
  // =========================================================================

  /**
   * Convert files to a more efficient format.
   *
   * @param sourcePath   Source S3 path
   * @param targetPath   Target S3 path
   * @param sourceFormat Source format (csv, json, etc.)
   * @param targetFormat Target format (parquet, delta)
   * @param partitionBy  Optional partition columns
   * @param compression  Compression codec
   * @return Conversion result
   *
   * Example:
   * {{{
   * optimizer.convertFormat(
   *   "s3://bucket/csv-data/",
   *   "s3://bucket/parquet-data/",
   *   "csv",
   *   "parquet",
   *   partitionBy = Some(Seq("date"))
   * )
   * }}}
   */
  def convertFormat(
      sourcePath: String,
      targetPath: String,
      sourceFormat: String,
      targetFormat: String = "parquet",
      partitionBy: Option[Seq[String]] = None,
      compression: String = "snappy"
  ): ConversionResult = {

    // Read source
    val reader = sourceFormat.toLowerCase match {
      case "csv" =>
        spark.read
          .option("header", "true")
          .option("inferSchema", "true")
          .format("csv")
      case "json" =>
        spark.read.format("json")
      case _ =>
        spark.read.format(sourceFormat)
    }

    val df = reader.load(sourcePath)
    val rowCount = df.count()

    // Write target
    var writer = df.write
      .format(targetFormat)
      .option("compression", compression)
      .mode("overwrite")

    partitionBy.foreach(cols => writer = writer.partitionBy(cols: _*))

    writer.save(targetPath)

    println(s"Converted $rowCount rows from $sourceFormat to $targetFormat")

    ConversionResult(
      sourcePath = sourcePath,
      targetPath = targetPath,
      sourceFormat = sourceFormat,
      targetFormat = targetFormat,
      rowCount = rowCount,
      partitionedBy = partitionBy
    )
  }

  // =========================================================================
  // COST OPTIMIZATION
  // =========================================================================

  /**
   * Estimate storage cost savings from format conversion.
   *
   * @param currentSizeGB   Current storage size in GB
   * @param currentFormat   Current format (csv, json)
   * @param targetFormat    Target format (parquet)
   * @return Estimated savings
   */
  def estimateSavings(
      currentSizeGB: Double,
      currentFormat: String,
      targetFormat: String = "parquet"
  ): Map[String, Double] = {

    // Compression ratios (approximate)
    val compressionRatios = Map(
      "csv_to_parquet" -> 0.25,
      "json_to_parquet" -> 0.20,
      "csv_to_delta" -> 0.25,
      "json_to_delta" -> 0.20
    )

    val key = s"${currentFormat.toLowerCase}_to_${targetFormat.toLowerCase}"
    val ratio = compressionRatios.getOrElse(key, 0.5)

    val newSizeGB = currentSizeGB * ratio
    val savedGB = currentSizeGB - newSizeGB

    // S3 pricing (approximate, us-east-1)
    val s3CostPerGB = 0.023 // per month

    Map(
      "currentSizeGB" -> currentSizeGB,
      "estimatedNewSizeGB" -> newSizeGB,
      "storageSavedGB" -> savedGB,
      "compressionRatio" -> ratio,
      "monthlySavingsUSD" -> savedGB * s3CostPerGB,
      "yearlySavingsUSD" -> savedGB * s3CostPerGB * 12
    )
  }

  /**
   * Get storage class recommendation based on access patterns.
   *
   * @param accessFrequency   Access frequency (frequent, infrequent, rare, archive)
   * @param dataCriticality   Data criticality (critical, important, low)
   * @return Recommended S3 storage class
   */
  def getStorageClassRecommendation(
      accessFrequency: String,
      dataCriticality: String
  ): String = {

    (accessFrequency.toLowerCase, dataCriticality.toLowerCase) match {
      case ("frequent", _) => "STANDARD"
      case ("infrequent", "critical") => "STANDARD_IA"
      case ("infrequent", _) => "ONEZONE_IA"
      case ("rare", _) => "GLACIER_INSTANT_RETRIEVAL"
      case ("archive", "critical") => "GLACIER"
      case ("archive", _) => "DEEP_ARCHIVE"
      case _ => "INTELLIGENT_TIERING"
    }
  }
}

/**
 * S3 analysis report.
 */
case class S3AnalysisReport(
    path: String,
    totalFiles: Long,
    totalSizeGB: Double,
    avgSizeMB: Double,
    minSizeMB: Double,
    maxSizeMB: Double,
    smallFileCount: Long,
    formats: Map[String, Long],
    issues: List[String],
    score: Int
) {
  override def toString: String = {
    s"""
       |S3 Analysis Report
       |==================
       |Path: $path
       |Total Files: $totalFiles
       |Total Size: ${f"$totalSizeGB%.2f"} GB
       |Avg File Size: ${f"$avgSizeMB%.1f"} MB
       |Min File Size: ${f"$minSizeMB%.1f"} MB
       |Max File Size: ${f"$maxSizeMB%.1f"} MB
       |Small Files (< 32MB): $smallFileCount
       |Formats: ${formats.map { case (k, v) => s"$k=$v" }.mkString(", ")}
       |Score: $score/100
       |
       |Issues:
       |${issues.map(i => s"  - $i").mkString("\n")}
       |""".stripMargin
  }
}

/**
 * File compaction result.
 */
case class CompactionResult(
    sourcePath: String,
    targetPath: String,
    inputFiles: Int,
    outputFiles: Int,
    format: String,
    compression: String
)

/**
 * Format conversion result.
 */
case class ConversionResult(
    sourcePath: String,
    targetPath: String,
    sourceFormat: String,
    targetFormat: String,
    rowCount: Long,
    partitionedBy: Option[Seq[String]]
)

/**
 * Companion object with utility methods.
 */
object S3Optimizer {

  /**
   * Create optimizer from active SparkSession.
   */
  def apply(): S3Optimizer = {
    new S3Optimizer(SparkSession.active)
  }

  /**
   * Recommended configurations for different workloads.
   */
  object Configs {
    val ConnectionMaximum = 200
    val MultipartThreshold = 64 * 1024 * 1024 // 64MB
    val FastUploadBuffer = "disk"
    val CommitterName = "magic"
  }

  /**
   * Optimal file size recommendations.
   */
  object FileSizes {
    val MinOptimalMB = 128
    val MaxOptimalMB = 1024
    val TargetMB = 256
  }
}
