/**
 * DataForge Unity Catalog Utilities for Scala
 *
 * Best practices for Unity Catalog management in Databricks.
 *
 * Key Concepts:
 *   - Three-level namespace: catalog.schema.table
 *   - Centralized governance and access control
 *   - Data lineage and audit logging
 *   - Cross-workspace data sharing
 *
 * Best Practices:
 *   1. Use meaningful catalog and schema names
 *   2. Implement consistent naming conventions
 *   3. Set appropriate permissions at each level
 *   4. Enable data lineage for compliance
 *   5. Document tables with comments
 *
 * @author DataForge Team
 */

package com.dataforge

import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.types.StructType

/**
 * Unity Catalog manager for Databricks.
 *
 * Provides utilities for managing catalogs, schemas, tables,
 * and permissions in Unity Catalog.
 *
 * Example usage:
 * {{{
 * val ucManager = new UnityCatalogManager(spark)
 *
 * // Create catalog and schema
 * ucManager.createCatalog("prod")
 * ucManager.createSchema("prod", "sales")
 *
 * // Create managed table
 * ucManager.createManagedTable(
 *   "prod.sales.orders",
 *   df,
 *   comment = "Customer orders table"
 * )
 *
 * // Grant permissions
 * ucManager.grantSelect("prod.sales.orders", "data_analysts")
 * }}}
 */
class UnityCatalogManager(spark: SparkSession) {

  import spark.implicits._

  // =========================================================================
  // CATALOG MANAGEMENT
  // =========================================================================

  /**
   * Create a new catalog.
   *
   * @param catalogName Name of the catalog
   * @param comment     Optional description
   *
   * Best Practice: Use meaningful names that reflect business domains.
   *
   * Example:
   * {{{
   * ucManager.createCatalog("prod", Some("Production data catalog"))
   * ucManager.createCatalog("dev", Some("Development sandbox"))
   * }}}
   */
  def createCatalog(catalogName: String, comment: Option[String] = None): Unit = {
    var sql = s"CREATE CATALOG IF NOT EXISTS $catalogName"
    comment.foreach(c => sql += s" COMMENT '$c'")

    println(s"Creating catalog: $catalogName")
    spark.sql(sql)
  }

  /**
   * Drop a catalog.
   *
   * @param catalogName Name of the catalog
   * @param cascade     If true, drop all schemas and tables
   *
   * Warning: CASCADE will delete all data in the catalog.
   */
  def dropCatalog(catalogName: String, cascade: Boolean = false): Unit = {
    val cascadeStr = if (cascade) " CASCADE" else ""
    val sql = s"DROP CATALOG IF EXISTS $catalogName$cascadeStr"

    println(s"Dropping catalog: $catalogName (cascade=$cascade)")
    spark.sql(sql)
  }

  /**
   * List all catalogs.
   *
   * @return DataFrame with catalog information
   */
  def listCatalogs(): DataFrame = {
    spark.sql("SHOW CATALOGS")
  }

  /**
   * Set current catalog.
   *
   * @param catalogName Catalog to use
   */
  def useCatalog(catalogName: String): Unit = {
    spark.sql(s"USE CATALOG $catalogName")
    println(s"Using catalog: $catalogName")
  }

  // =========================================================================
  // SCHEMA MANAGEMENT
  // =========================================================================

  /**
   * Create a new schema.
   *
   * @param catalogName Catalog name
   * @param schemaName  Schema name
   * @param comment     Optional description
   * @param location    Optional managed location
   *
   * Best Practice: Organize schemas by business domain or data source.
   *
   * Example:
   * {{{
   * ucManager.createSchema("prod", "sales", Some("Sales department data"))
   * ucManager.createSchema("prod", "marketing", Some("Marketing analytics"))
   * }}}
   */
  def createSchema(
      catalogName: String,
      schemaName: String,
      comment: Option[String] = None,
      location: Option[String] = None
  ): Unit = {

    var sql = s"CREATE SCHEMA IF NOT EXISTS $catalogName.$schemaName"
    comment.foreach(c => sql += s" COMMENT '$c'")
    location.foreach(l => sql += s" MANAGED LOCATION '$l'")

    println(s"Creating schema: $catalogName.$schemaName")
    spark.sql(sql)
  }

  /**
   * Drop a schema.
   *
   * @param catalogName Catalog name
   * @param schemaName  Schema name
   * @param cascade     If true, drop all tables
   */
  def dropSchema(
      catalogName: String,
      schemaName: String,
      cascade: Boolean = false
  ): Unit = {
    val cascadeStr = if (cascade) " CASCADE" else ""
    val sql = s"DROP SCHEMA IF EXISTS $catalogName.$schemaName$cascadeStr"

    println(s"Dropping schema: $catalogName.$schemaName (cascade=$cascade)")
    spark.sql(sql)
  }

  /**
   * List schemas in a catalog.
   *
   * @param catalogName Catalog name
   * @return DataFrame with schema information
   */
  def listSchemas(catalogName: String): DataFrame = {
    spark.sql(s"SHOW SCHEMAS IN $catalogName")
  }

  /**
   * Set current schema.
   *
   * @param schemaName Schema to use
   */
  def useSchema(schemaName: String): Unit = {
    spark.sql(s"USE SCHEMA $schemaName")
    println(s"Using schema: $schemaName")
  }

  // =========================================================================
  // TABLE MANAGEMENT
  // =========================================================================

  /**
   * Create a managed Delta table.
   *
   * @param fullTableName Full table name (catalog.schema.table)
   * @param df            DataFrame with data
   * @param comment       Optional table description
   * @param partitionBy   Optional partition columns
   * @param properties    Optional table properties
   *
   * Best Practice:
   *   - Always add meaningful comments
   *   - Partition large tables by date/region
   *   - Use appropriate table properties
   *
   * Example:
   * {{{
   * ucManager.createManagedTable(
   *   "prod.sales.orders",
   *   ordersDF,
   *   comment = Some("Customer orders"),
   *   partitionBy = Some(Seq("order_date")),
   *   properties = Map("delta.autoOptimize.optimizeWrite" -> "true")
   * )
   * }}}
   */
  def createManagedTable(
      fullTableName: String,
      df: DataFrame,
      comment: Option[String] = None,
      partitionBy: Option[Seq[String]] = None,
      properties: Map[String, String] = Map.empty
  ): Unit = {

    var writer = df.write
      .format("delta")
      .mode("overwrite")

    partitionBy.foreach(cols => writer = writer.partitionBy(cols: _*))
    properties.foreach { case (k, v) => writer = writer.option(k, v) }

    println(s"Creating managed table: $fullTableName")
    writer.saveAsTable(fullTableName)

    // Add comment if provided
    comment.foreach { c =>
      spark.sql(s"COMMENT ON TABLE $fullTableName IS '$c'")
    }

    // Set properties
    if (properties.nonEmpty) {
      val propsStr = properties.map { case (k, v) => s"'$k' = '$v'" }.mkString(", ")
      spark.sql(s"ALTER TABLE $fullTableName SET TBLPROPERTIES ($propsStr)")
    }
  }

  /**
   * Create an external table.
   *
   * @param fullTableName Full table name
   * @param location      External storage location
   * @param schema        Table schema
   * @param format        File format (delta, parquet, csv)
   * @param comment       Optional description
   */
  def createExternalTable(
      fullTableName: String,
      location: String,
      schema: Option[StructType] = None,
      format: String = "delta",
      comment: Option[String] = None
  ): Unit = {

    var sql = s"CREATE TABLE IF NOT EXISTS $fullTableName"

    schema.foreach { s =>
      val schemaStr = s.fields.map(f => s"${f.name} ${f.dataType.sql}").mkString(", ")
      sql += s" ($schemaStr)"
    }

    sql += s" USING $format LOCATION '$location'"

    println(s"Creating external table: $fullTableName at $location")
    spark.sql(sql)

    comment.foreach { c =>
      spark.sql(s"COMMENT ON TABLE $fullTableName IS '$c'")
    }
  }

  /**
   * Drop a table.
   *
   * @param fullTableName Full table name
   */
  def dropTable(fullTableName: String): Unit = {
    spark.sql(s"DROP TABLE IF EXISTS $fullTableName")
    println(s"Dropped table: $fullTableName")
  }

  /**
   * List tables in a schema.
   *
   * @param catalogName Catalog name
   * @param schemaName  Schema name
   * @return DataFrame with table information
   */
  def listTables(catalogName: String, schemaName: String): DataFrame = {
    spark.sql(s"SHOW TABLES IN $catalogName.$schemaName")
  }

  /**
   * Get table details.
   *
   * @param fullTableName Full table name
   * @return DataFrame with table details
   */
  def describeTable(fullTableName: String): DataFrame = {
    spark.sql(s"DESCRIBE TABLE EXTENDED $fullTableName")
  }

  /**
   * Add or update table comment.
   *
   * @param fullTableName Full table name
   * @param comment       Description
   */
  def setTableComment(fullTableName: String, comment: String): Unit = {
    spark.sql(s"COMMENT ON TABLE $fullTableName IS '$comment'")
    println(s"Updated comment for $fullTableName")
  }

  /**
   * Add column comments.
   *
   * @param fullTableName Full table name
   * @param comments      Map of column name to comment
   *
   * Example:
   * {{{
   * ucManager.setColumnComments("prod.sales.orders", Map(
   *   "order_id" -> "Unique order identifier",
   *   "customer_id" -> "Reference to customers table",
   *   "total_amount" -> "Order total in USD"
   * ))
   * }}}
   */
  def setColumnComments(fullTableName: String, comments: Map[String, String]): Unit = {
    comments.foreach { case (column, comment) =>
      spark.sql(s"ALTER TABLE $fullTableName ALTER COLUMN $column COMMENT '$comment'")
    }
    println(s"Updated column comments for $fullTableName")
  }

  // =========================================================================
  // PERMISSIONS MANAGEMENT
  // =========================================================================

  /**
   * Grant SELECT permission on a table.
   *
   * @param fullTableName Full table name
   * @param principal     User, group, or service principal
   *
   * Example:
   * {{{
   * ucManager.grantSelect("prod.sales.orders", "data_analysts")
   * }}}
   */
  def grantSelect(fullTableName: String, principal: String): Unit = {
    spark.sql(s"GRANT SELECT ON TABLE $fullTableName TO `$principal`")
    println(s"Granted SELECT on $fullTableName to $principal")
  }

  /**
   * Grant multiple permissions on a table.
   *
   * @param fullTableName Full table name
   * @param principal     User, group, or service principal
   * @param permissions   Permissions to grant (SELECT, MODIFY, etc.)
   */
  def grantPermissions(
      fullTableName: String,
      principal: String,
      permissions: Seq[String]
  ): Unit = {
    permissions.foreach { perm =>
      spark.sql(s"GRANT $perm ON TABLE $fullTableName TO `$principal`")
    }
    println(s"Granted ${permissions.mkString(", ")} on $fullTableName to $principal")
  }

  /**
   * Revoke permission on a table.
   *
   * @param fullTableName Full table name
   * @param principal     User, group, or service principal
   * @param permission    Permission to revoke
   */
  def revokePermission(
      fullTableName: String,
      principal: String,
      permission: String
  ): Unit = {
    spark.sql(s"REVOKE $permission ON TABLE $fullTableName FROM `$principal`")
    println(s"Revoked $permission on $fullTableName from $principal")
  }

  /**
   * Grant schema-level permissions.
   *
   * @param catalogName  Catalog name
   * @param schemaName   Schema name
   * @param principal    User, group, or service principal
   * @param permissions  Permissions to grant
   */
  def grantSchemaPermissions(
      catalogName: String,
      schemaName: String,
      principal: String,
      permissions: Seq[String]
  ): Unit = {
    permissions.foreach { perm =>
      spark.sql(s"GRANT $perm ON SCHEMA $catalogName.$schemaName TO `$principal`")
    }
    println(s"Granted schema permissions on $catalogName.$schemaName to $principal")
  }

  /**
   * Show grants on a securable object.
   *
   * @param securableType Type (TABLE, SCHEMA, CATALOG)
   * @param securableName Name of the object
   * @return DataFrame with grant information
   */
  def showGrants(securableType: String, securableName: String): DataFrame = {
    spark.sql(s"SHOW GRANTS ON $securableType $securableName")
  }

  // =========================================================================
  // DATA LINEAGE
  // =========================================================================

  /**
   * Get table lineage information.
   *
   * Note: Requires Unity Catalog with lineage enabled.
   *
   * @param fullTableName Full table name
   * @return DataFrame with lineage information
   */
  def getTableLineage(fullTableName: String): DataFrame = {
    // Lineage is available through system tables
    spark.sql(s"""
      SELECT *
      FROM system.access.table_lineage
      WHERE target_table_full_name = '$fullTableName'
      ORDER BY event_time DESC
      LIMIT 100
    """)
  }

  /**
   * Get column lineage information.
   *
   * @param fullTableName Full table name
   * @param columnName    Column name
   * @return DataFrame with column lineage
   */
  def getColumnLineage(fullTableName: String, columnName: String): DataFrame = {
    spark.sql(s"""
      SELECT *
      FROM system.access.column_lineage
      WHERE target_table_full_name = '$fullTableName'
        AND target_column_name = '$columnName'
      ORDER BY event_time DESC
      LIMIT 100
    """)
  }

  // =========================================================================
  // VIEWS
  // =========================================================================

  /**
   * Create a view.
   *
   * @param viewName Full view name (catalog.schema.view)
   * @param query    SELECT query for the view
   * @param comment  Optional description
   *
   * Best Practice: Use views to simplify complex queries and control access.
   *
   * Example:
   * {{{
   * ucManager.createView(
   *   "prod.sales.orders_summary",
   *   "SELECT customer_id, SUM(total) as total_spent FROM prod.sales.orders GROUP BY customer_id",
   *   Some("Aggregated customer spending")
   * )
   * }}}
   */
  def createView(
      viewName: String,
      query: String,
      comment: Option[String] = None
  ): Unit = {
    spark.sql(s"CREATE OR REPLACE VIEW $viewName AS $query")
    println(s"Created view: $viewName")

    comment.foreach { c =>
      spark.sql(s"COMMENT ON VIEW $viewName IS '$c'")
    }
  }

  /**
   * Drop a view.
   *
   * @param viewName Full view name
   */
  def dropView(viewName: String): Unit = {
    spark.sql(s"DROP VIEW IF EXISTS $viewName")
    println(s"Dropped view: $viewName")
  }

  // =========================================================================
  // FUNCTIONS
  // =========================================================================

  /**
   * Register a UDF in Unity Catalog.
   *
   * @param functionName Full function name (catalog.schema.function)
   * @param className    Full class name implementing the UDF
   * @param comment      Optional description
   */
  def registerFunction(
      functionName: String,
      className: String,
      comment: Option[String] = None
  ): Unit = {
    spark.sql(s"CREATE OR REPLACE FUNCTION $functionName AS '$className'")
    println(s"Registered function: $functionName")

    comment.foreach { c =>
      spark.sql(s"COMMENT ON FUNCTION $functionName IS '$c'")
    }
  }

  /**
   * List functions in a schema.
   *
   * @param catalogName Catalog name
   * @param schemaName  Schema name
   * @return DataFrame with function information
   */
  def listFunctions(catalogName: String, schemaName: String): DataFrame = {
    spark.sql(s"SHOW FUNCTIONS IN $catalogName.$schemaName")
  }

  // =========================================================================
  // VOLUMES (External Data)
  // =========================================================================

  /**
   * Create a volume for external file storage.
   *
   * Volumes provide governed access to external files.
   *
   * @param catalogName Catalog name
   * @param schemaName  Schema name
   * @param volumeName  Volume name
   * @param location    External location
   * @param comment     Optional description
   *
   * Example:
   * {{{
   * ucManager.createVolume(
   *   "prod", "raw_data", "landing_zone",
   *   "s3://bucket/landing/",
   *   Some("Raw data landing zone")
   * )
   * }}}
   */
  def createVolume(
      catalogName: String,
      schemaName: String,
      volumeName: String,
      location: String,
      comment: Option[String] = None
  ): Unit = {
    var sql = s"CREATE EXTERNAL VOLUME IF NOT EXISTS $catalogName.$schemaName.$volumeName"
    sql += s" LOCATION '$location'"
    comment.foreach(c => sql += s" COMMENT '$c'")

    println(s"Creating volume: $catalogName.$schemaName.$volumeName")
    spark.sql(sql)
  }

  /**
   * List volumes in a schema.
   *
   * @param catalogName Catalog name
   * @param schemaName  Schema name
   * @return DataFrame with volume information
   */
  def listVolumes(catalogName: String, schemaName: String): DataFrame = {
    spark.sql(s"SHOW VOLUMES IN $catalogName.$schemaName")
  }

  // =========================================================================
  // UTILITY METHODS
  // =========================================================================

  /**
   * Parse full table name into components.
   *
   * @param fullTableName Full table name (catalog.schema.table)
   * @return Tuple of (catalog, schema, table)
   */
  def parseTableName(fullTableName: String): (String, String, String) = {
    val parts = fullTableName.split("\\.")
    if (parts.length != 3) {
      throw new IllegalArgumentException(
        s"Invalid table name: $fullTableName. Expected format: catalog.schema.table"
      )
    }
    (parts(0), parts(1), parts(2))
  }

  /**
   * Build full table name from components.
   *
   * @param catalog Catalog name
   * @param schema  Schema name
   * @param table   Table name
   * @return Full table name
   */
  def buildTableName(catalog: String, schema: String, table: String): String = {
    s"$catalog.$schema.$table"
  }

  /**
   * Check if a table exists.
   *
   * @param fullTableName Full table name
   * @return true if table exists
   */
  def tableExists(fullTableName: String): Boolean = {
    try {
      spark.sql(s"DESCRIBE TABLE $fullTableName")
      true
    } catch {
      case _: Exception => false
    }
  }

  /**
   * Get current catalog.
   *
   * @return Current catalog name
   */
  def currentCatalog(): String = {
    spark.sql("SELECT current_catalog()").collect()(0).getString(0)
  }

  /**
   * Get current schema.
   *
   * @return Current schema name
   */
  def currentSchema(): String = {
    spark.sql("SELECT current_schema()").collect()(0).getString(0)
  }
}

/**
 * Companion object with utility methods and constants.
 */
object UnityCatalogManager {

  /**
   * Create manager from active SparkSession.
   */
  def apply(): UnityCatalogManager = {
    new UnityCatalogManager(SparkSession.active)
  }

  /**
   * Standard permission levels.
   */
  object Permissions {
    val SELECT = "SELECT"
    val MODIFY = "MODIFY"
    val CREATE = "CREATE"
    val USAGE = "USAGE"
    val READ_METADATA = "READ_METADATA"
    val CREATE_TABLE = "CREATE TABLE"
    val CREATE_VIEW = "CREATE VIEW"
    val CREATE_FUNCTION = "CREATE FUNCTION"
    val ALL_PRIVILEGES = "ALL PRIVILEGES"
  }

  /**
   * Naming convention recommendations.
   */
  object NamingConventions {
    val CatalogPattern = "^[a-z][a-z0-9_]*$".r
    val SchemaPattern = "^[a-z][a-z0-9_]*$".r
    val TablePattern = "^[a-z][a-z0-9_]*$".r

    def validateCatalogName(name: String): Boolean = CatalogPattern.matches(name)
    def validateSchemaName(name: String): Boolean = SchemaPattern.matches(name)
    def validateTableName(name: String): Boolean = TablePattern.matches(name)
  }
}
