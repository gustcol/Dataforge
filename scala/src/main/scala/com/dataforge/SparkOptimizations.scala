/**
 * DataForge Spark Optimizations for Scala
 *
 * Best practices for Apache Spark configuration and optimization
 * in Databricks environments.
 *
 * Key Areas:
 *   - Adaptive Query Execution (AQE)
 *   - Broadcast joins
 *   - Partition management
 *   - Memory configuration
 *   - Shuffle optimization
 *
 * Best Practices:
 *   1. Enable AQE for automatic query optimization
 *   2. Use broadcast joins for small tables (< 10MB)
 *   3. Configure appropriate shuffle partitions
 *   4. Optimize memory allocation for executors
 *   5. Use columnar formats (Parquet, Delta)
 *
 * @author DataForge Team
 */

package com.dataforge

import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.storage.StorageLevel

/**
 * Spark configuration optimizer for Databricks workloads.
 *
 * Provides methods to configure Spark for optimal performance
 * based on workload characteristics.
 *
 * Example usage:
 * {{{
 * val optimizer = new SparkOptimizer(spark)
 *
 * // Apply recommended configurations
 * optimizer.applyRecommendedConfig()
 *
 * // Configure for ETL workload
 * optimizer.configureForETL(dataSize = "large")
 *
 * // Optimize specific DataFrame operations
 * val optimizedDf = optimizer.optimizeJoin(leftDf, rightDf, "key")
 * }}}
 */
class SparkOptimizer(spark: SparkSession) {

  import spark.implicits._

  // =========================================================================
  // ADAPTIVE QUERY EXECUTION (AQE)
  // =========================================================================

  /**
   * Enable Adaptive Query Execution with optimal settings.
   *
   * AQE dynamically optimizes query execution based on runtime statistics:
   * - Coalesces shuffle partitions
   * - Converts sort-merge joins to broadcast joins
   * - Optimizes skewed joins
   *
   * Best Practice: Always enable AQE in Databricks.
   */
  def enableAQE(): Unit = {
    spark.conf.set("spark.sql.adaptive.enabled", "true")
    spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")
    spark.conf.set("spark.sql.adaptive.skewJoin.enabled", "true")
    spark.conf.set("spark.sql.adaptive.localShuffleReader.enabled", "true")
    println("AQE enabled with all optimizations")
  }

  /**
   * Configure AQE partition coalescing.
   *
   * @param targetSize Target partition size after coalescing (default: 64MB)
   * @param minPartitions Minimum number of partitions to maintain
   */
  def configurePartitionCoalescing(
      targetSize: String = "64MB",
      minPartitions: Int = 1
  ): Unit = {
    spark.conf.set("spark.sql.adaptive.advisoryPartitionSizeInBytes", targetSize)
    spark.conf.set("spark.sql.adaptive.coalescePartitions.minPartitionNum", minPartitions)
    println(s"Partition coalescing configured: target=$targetSize, min=$minPartitions")
  }

  /**
   * Configure skew join optimization.
   *
   * @param skewedPartitionFactor Factor to identify skewed partitions
   * @param skewedPartitionThreshold Minimum size for skewed partition
   */
  def configureSkewJoin(
      skewedPartitionFactor: Int = 5,
      skewedPartitionThreshold: String = "256MB"
  ): Unit = {
    spark.conf.set("spark.sql.adaptive.skewJoin.skewedPartitionFactor", skewedPartitionFactor)
    spark.conf.set("spark.sql.adaptive.skewJoin.skewedPartitionThresholdInBytes", skewedPartitionThreshold)
    println(s"Skew join configured: factor=$skewedPartitionFactor, threshold=$skewedPartitionThreshold")
  }

  // =========================================================================
  // BROADCAST JOINS
  // =========================================================================

  /**
   * Configure broadcast join threshold.
   *
   * Tables smaller than this threshold will be broadcast to all executors.
   *
   * @param thresholdMB Threshold in megabytes (default: 10MB)
   *
   * Best Practice:
   *   - Set to 10-100MB depending on executor memory
   *   - Monitor broadcast time vs shuffle time
   *   - Use explicit broadcast() for critical joins
   */
  def setBroadcastThreshold(thresholdMB: Int = 10): Unit = {
    val bytes = thresholdMB * 1024 * 1024
    spark.conf.set("spark.sql.autoBroadcastJoinThreshold", bytes)
    println(s"Broadcast threshold set to ${thresholdMB}MB")
  }

  /**
   * Disable automatic broadcast joins.
   *
   * Useful when you want full control over join strategies.
   */
  def disableAutoBroadcast(): Unit = {
    spark.conf.set("spark.sql.autoBroadcastJoinThreshold", -1)
    println("Auto broadcast joins disabled")
  }

  /**
   * Perform optimized join with automatic strategy selection.
   *
   * @param leftDf    Left DataFrame
   * @param rightDf   Right DataFrame
   * @param joinKeys  Join key columns
   * @param joinType  Type of join (inner, left, right, outer)
   * @return Joined DataFrame
   *
   * Example:
   * {{{
   * val result = optimizer.optimizeJoin(
   *   orders,
   *   customers,
   *   Seq("customer_id"),
   *   "left"
   * )
   * }}}
   */
  def optimizeJoin(
      leftDf: DataFrame,
      rightDf: DataFrame,
      joinKeys: Seq[String],
      joinType: String = "inner"
  ): DataFrame = {

    // Estimate sizes (rough heuristic)
    val leftSize = estimateSizeMB(leftDf)
    val rightSize = estimateSizeMB(rightDf)
    val broadcastThreshold = 100 // MB

    val result = (leftSize, rightSize) match {
      case (l, r) if r < broadcastThreshold && r < l =>
        // Broadcast right side
        println(s"Using broadcast join (right side: ~${rightSize}MB)")
        leftDf.join(broadcast(rightDf), joinKeys, joinType)

      case (l, r) if l < broadcastThreshold && l < r =>
        // Broadcast left side
        println(s"Using broadcast join (left side: ~${leftSize}MB)")
        broadcast(leftDf).join(rightDf, joinKeys, joinType)

      case _ =>
        // Use sort-merge join (let AQE optimize)
        println("Using sort-merge join with AQE optimization")
        leftDf.join(rightDf, joinKeys, joinType)
    }

    result
  }

  // =========================================================================
  // SHUFFLE OPTIMIZATION
  // =========================================================================

  /**
   * Configure shuffle partitions.
   *
   * @param partitions Number of shuffle partitions
   *
   * Guidelines:
   *   - Small data (< 1GB): 20-50 partitions
   *   - Medium data (1-10GB): 100-200 partitions
   *   - Large data (10-100GB): 200-500 partitions
   *   - Very large data (> 100GB): 500-2000 partitions
   *
   * Best Practice: Use AQE for automatic partition coalescing.
   */
  def setShufflePartitions(partitions: Int): Unit = {
    spark.conf.set("spark.sql.shuffle.partitions", partitions)
    println(s"Shuffle partitions set to $partitions")
  }

  /**
   * Calculate optimal shuffle partitions based on data size.
   *
   * @param dataSizeMB Estimated data size in MB
   * @param targetPartitionSizeMB Target partition size (default: 128MB)
   * @return Recommended partition count
   */
  def calculateOptimalPartitions(
      dataSizeMB: Double,
      targetPartitionSizeMB: Int = 128
  ): Int = {
    val partitions = math.max(1, math.ceil(dataSizeMB / targetPartitionSizeMB).toInt)
    println(s"Recommended partitions for ${dataSizeMB}MB data: $partitions")
    partitions
  }

  /**
   * Enable shuffle service for better shuffle performance.
   */
  def enableShuffleService(): Unit = {
    spark.conf.set("spark.shuffle.service.enabled", "true")
    println("External shuffle service enabled")
  }

  // =========================================================================
  // MEMORY CONFIGURATION
  // =========================================================================

  /**
   * Configure memory settings for optimal performance.
   *
   * @param memoryFraction Fraction of heap for execution/storage (default: 0.6)
   * @param storageFraction Fraction of memory pool for storage (default: 0.5)
   */
  def configureMemory(
      memoryFraction: Double = 0.6,
      storageFraction: Double = 0.5
  ): Unit = {
    spark.conf.set("spark.memory.fraction", memoryFraction)
    spark.conf.set("spark.memory.storageFraction", storageFraction)
    println(s"Memory configured: fraction=$memoryFraction, storage=$storageFraction")
  }

  /**
   * Configure off-heap memory.
   *
   * @param enabled     Enable off-heap memory
   * @param sizeGB      Off-heap memory size in GB
   *
   * Best Practice: Use off-heap for large aggregations.
   */
  def configureOffHeap(enabled: Boolean = true, sizeGB: Int = 4): Unit = {
    spark.conf.set("spark.memory.offHeap.enabled", enabled)
    if (enabled) {
      spark.conf.set("spark.memory.offHeap.size", s"${sizeGB}g")
    }
    println(s"Off-heap memory: enabled=$enabled, size=${sizeGB}GB")
  }

  // =========================================================================
  // CACHING STRATEGIES
  // =========================================================================

  /**
   * Cache DataFrame with optimal storage level.
   *
   * @param df          DataFrame to cache
   * @param level       Storage level (MEMORY_AND_DISK by default)
   * @param tableName   Optional table name for cache identification
   * @return Cached DataFrame
   *
   * Best Practices:
   *   - Only cache DataFrames used multiple times
   *   - Use MEMORY_AND_DISK for large DataFrames
   *   - Always unpersist when done
   */
  def cacheOptimized(
      df: DataFrame,
      level: StorageLevel = StorageLevel.MEMORY_AND_DISK,
      tableName: Option[String] = None
  ): DataFrame = {
    val cached = df.persist(level)

    // Force caching
    cached.count()

    tableName.foreach(name => println(s"Cached DataFrame: $name"))
    println(s"Storage level: $level")

    cached
  }

  /**
   * Unpersist DataFrame and release memory.
   *
   * @param df      DataFrame to unpersist
   * @param blocking Wait for unpersist to complete
   */
  def unpersistSafe(df: DataFrame, blocking: Boolean = true): Unit = {
    df.unpersist(blocking)
    println("DataFrame unpersisted")
  }

  /**
   * Clear all cached data.
   *
   * Use after completing a processing stage to free memory.
   */
  def clearAllCache(): Unit = {
    spark.catalog.clearCache()
    println("All cache cleared")
  }

  // =========================================================================
  // WORKLOAD-SPECIFIC CONFIGURATIONS
  // =========================================================================

  /**
   * Apply recommended configuration for all workloads.
   *
   * Enables common optimizations that benefit most scenarios.
   */
  def applyRecommendedConfig(): Unit = {
    // Enable AQE
    enableAQE()

    // Set reasonable defaults
    setBroadcastThreshold(10)
    configurePartitionCoalescing()
    configureMemory()

    // Enable optimizations
    spark.conf.set("spark.sql.files.maxPartitionBytes", "128m")
    spark.conf.set("spark.sql.parquet.mergeSchema", "false")
    spark.conf.set("spark.sql.parquet.filterPushdown", "true")
    spark.conf.set("spark.sql.inMemoryColumnarStorage.compressed", "true")

    println("Recommended configuration applied")
  }

  /**
   * Configure for ETL workloads.
   *
   * @param dataSize Size category: "small", "medium", "large", "xlarge"
   */
  def configureForETL(dataSize: String): Unit = {
    applyRecommendedConfig()

    dataSize.toLowerCase match {
      case "small" =>
        setShufflePartitions(50)
        setBroadcastThreshold(50)

      case "medium" =>
        setShufflePartitions(200)
        setBroadcastThreshold(30)

      case "large" =>
        setShufflePartitions(500)
        setBroadcastThreshold(10)
        configureOffHeap(enabled = true, sizeGB = 4)

      case "xlarge" =>
        setShufflePartitions(2000)
        setBroadcastThreshold(10)
        configureOffHeap(enabled = true, sizeGB = 8)

      case _ =>
        println(s"Unknown data size: $dataSize, using defaults")
    }

    println(s"ETL configuration applied for $dataSize data")
  }

  /**
   * Configure for streaming workloads.
   *
   * @param checkpointLocation Checkpoint directory
   * @param triggerInterval   Trigger interval (e.g., "10 seconds")
   */
  def configureForStreaming(
      checkpointLocation: String,
      triggerInterval: String = "10 seconds"
  ): Unit = {
    applyRecommendedConfig()

    // Streaming-specific settings
    spark.conf.set("spark.sql.streaming.checkpointLocation", checkpointLocation)
    spark.conf.set("spark.sql.shuffle.partitions", "100")

    // Reduce latency
    spark.conf.set("spark.sql.adaptive.enabled", "false") // Can cause latency spikes

    println(s"Streaming configuration applied: checkpoint=$checkpointLocation")
  }

  /**
   * Configure for ML workloads.
   *
   * ML workloads often require more memory and specific optimizations.
   */
  def configureForML(): Unit = {
    applyRecommendedConfig()

    // ML-specific settings
    configureMemory(memoryFraction = 0.7, storageFraction = 0.4)
    configureOffHeap(enabled = true, sizeGB = 4)
    setShufflePartitions(200)

    // Enable BLAS optimization if available
    spark.conf.set("spark.ml.linalg.useBLAS", "true")

    println("ML configuration applied")
  }

  /**
   * Configure for interactive/BI workloads.
   *
   * Prioritizes query latency over throughput.
   */
  def configureForInteractive(): Unit = {
    applyRecommendedConfig()

    // Favor low latency
    setShufflePartitions(50)
    setBroadcastThreshold(100) // More aggressive broadcasting

    // Enable result caching
    spark.conf.set("spark.sql.inMemoryColumnarStorage.batchSize", "20000")

    println("Interactive configuration applied")
  }

  // =========================================================================
  // MONITORING AND DIAGNOSTICS
  // =========================================================================

  /**
   * Get current Spark configuration.
   *
   * @return Map of configuration key-value pairs
   */
  def getCurrentConfig: Map[String, String] = {
    spark.conf.getAll.toMap
  }

  /**
   * Print key configuration values.
   */
  def printConfig(): Unit = {
    val keyConfigs = Seq(
      "spark.sql.adaptive.enabled",
      "spark.sql.shuffle.partitions",
      "spark.sql.autoBroadcastJoinThreshold",
      "spark.memory.fraction",
      "spark.memory.storageFraction",
      "spark.memory.offHeap.enabled",
      "spark.memory.offHeap.size"
    )

    println("\n=== Spark Configuration ===")
    keyConfigs.foreach { key =>
      val value = spark.conf.getOption(key).getOrElse("not set")
      println(s"$key = $value")
    }
    println("===========================\n")
  }

  /**
   * Explain query execution plan.
   *
   * @param df    DataFrame to explain
   * @param mode  Explain mode: "simple", "extended", "codegen", "cost", "formatted"
   */
  def explainQuery(df: DataFrame, mode: String = "formatted"): Unit = {
    println(s"\n=== Query Plan ($mode) ===")
    df.explain(mode)
    println("==========================\n")
  }

  // =========================================================================
  // HELPER METHODS
  // =========================================================================

  /**
   * Estimate DataFrame size in MB (rough approximation).
   */
  private def estimateSizeMB(df: DataFrame): Double = {
    try {
      val sampleSize = 1000
      val sample = df.limit(sampleSize).cache()
      val sampleCount = sample.count()

      if (sampleCount == 0) {
        sample.unpersist()
        return 0.0
      }

      // Get sample stats
      val stats = spark.sessionState.executePlan(sample.queryExecution.logical).optimizedPlan.stats
      val sampleBytes = stats.sizeInBytes.toLong

      sample.unpersist()

      // Extrapolate to full size
      val totalCount = df.count()
      val estimatedBytes = (sampleBytes.toDouble / sampleCount) * totalCount
      estimatedBytes / (1024.0 * 1024.0)
    } catch {
      case _: Exception =>
        // Fallback: rough estimate based on schema
        val numCols = df.columns.length
        val numRows = df.count()
        (numCols * numRows * 50.0) / (1024.0 * 1024.0)
    }
  }
}

/**
 * Companion object with utility methods and constants.
 */
object SparkOptimizer {

  /**
   * Create optimizer from active SparkSession.
   */
  def apply(): SparkOptimizer = {
    new SparkOptimizer(SparkSession.active)
  }

  /**
   * Default configuration values.
   */
  object Defaults {
    val ShufflePartitions: Int = 200
    val BroadcastThresholdMB: Int = 10
    val TargetPartitionSizeMB: Int = 128
    val MemoryFraction: Double = 0.6
    val StorageFraction: Double = 0.5
  }

  /**
   * Storage level recommendations.
   */
  object StorageLevels {
    val ForSmallData: StorageLevel = StorageLevel.MEMORY_ONLY
    val ForMediumData: StorageLevel = StorageLevel.MEMORY_AND_DISK
    val ForLargeData: StorageLevel = StorageLevel.MEMORY_AND_DISK_SER
    val ForReplication: StorageLevel = StorageLevel.MEMORY_AND_DISK_2
  }
}
