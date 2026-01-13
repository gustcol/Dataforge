"""
DataForge Unity Catalog Manager

Unity Catalog is Databricks' unified governance solution for all data and AI assets.
This module provides utilities for managing Unity Catalog resources.

Three-Level Namespace:
    Unity Catalog uses a three-level namespace: catalog.schema.table
    - Catalog: Top-level container, often per business unit or environment
    - Schema: Logical grouping of tables (similar to database)
    - Table: The actual data table

Features:
    - Catalog and schema management
    - Table and view management
    - Permission management (GRANT/REVOKE)
    - Data lineage tracking
    - Tagging and metadata management

Best Practices:
    1. Use separate catalogs for dev/staging/prod
    2. Organize schemas by domain or team
    3. Apply least-privilege permissions
    4. Use managed tables for better governance
    5. Leverage tags for data classification

Example:
    >>> from dataforge.databricks import UnityCatalogManager
    >>>
    >>> uc = UnityCatalogManager(spark)
    >>>
    >>> # Create catalog and schema
    >>> uc.create_catalog("analytics")
    >>> uc.create_schema("analytics", "sales")
    >>>
    >>> # List tables
    >>> tables = uc.list_tables("analytics", "sales")
    >>>
    >>> # Grant permissions
    >>> uc.grant_select("analytics.sales.orders", "data_analysts")
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TYPE_CHECKING
import logging

if TYPE_CHECKING:
    from pyspark.sql import SparkSession

logger = logging.getLogger(__name__)


@dataclass
class CatalogInfo:
    """Information about a Unity Catalog."""
    name: str
    comment: Optional[str] = None
    owner: Optional[str] = None
    created_at: Optional[str] = None
    properties: Optional[Dict[str, str]] = None


@dataclass
class SchemaInfo:
    """Information about a schema in Unity Catalog."""
    name: str
    catalog_name: str
    comment: Optional[str] = None
    owner: Optional[str] = None
    properties: Optional[Dict[str, str]] = None


@dataclass
class TableInfo:
    """Information about a table in Unity Catalog."""
    name: str
    schema_name: str
    catalog_name: str
    table_type: str  # MANAGED, EXTERNAL, VIEW
    data_source_format: Optional[str] = None
    comment: Optional[str] = None
    owner: Optional[str] = None
    location: Optional[str] = None
    columns: Optional[List[Dict[str, str]]] = None


class UnityCatalogManager:
    """
    Manager for Unity Catalog operations.

    Provides methods for managing catalogs, schemas, tables, and permissions
    in Databricks Unity Catalog.

    Requirements:
        - Databricks Runtime 11.3+
        - Unity Catalog enabled workspace
        - Appropriate permissions (CREATE CATALOG, etc.)

    Example:
        >>> uc = UnityCatalogManager(spark)
        >>>
        >>> # Catalog operations
        >>> uc.create_catalog("my_catalog", comment="Production data")
        >>> catalogs = uc.list_catalogs()
        >>>
        >>> # Schema operations
        >>> uc.create_schema("my_catalog", "bronze", comment="Raw data layer")
        >>> uc.create_schema("my_catalog", "silver", comment="Cleansed data")
        >>> uc.create_schema("my_catalog", "gold", comment="Business aggregates")
        >>>
        >>> # Permission operations
        >>> uc.grant_select("my_catalog.bronze.events", "data_engineers")
    """

    def __init__(self, spark: "SparkSession") -> None:
        """
        Initialize Unity Catalog Manager.

        Args:
            spark: SparkSession with Unity Catalog access
        """
        self.spark = spark

    # =========================================================================
    # CATALOG OPERATIONS
    # =========================================================================

    def create_catalog(
        self,
        name: str,
        comment: Optional[str] = None,
        if_not_exists: bool = True
    ) -> None:
        """
        Create a new catalog.

        Args:
            name: Catalog name
            comment: Optional description
            if_not_exists: Don't error if exists

        Example:
            >>> uc.create_catalog("production", comment="Production data catalog")
        """
        exists_clause = "IF NOT EXISTS " if if_not_exists else ""
        comment_clause = f" COMMENT '{comment}'" if comment else ""

        sql = f"CREATE CATALOG {exists_clause}{name}{comment_clause}"
        self.spark.sql(sql)
        logger.info(f"Created catalog: {name}")

    def drop_catalog(
        self,
        name: str,
        cascade: bool = False,
        if_exists: bool = True
    ) -> None:
        """
        Drop a catalog.

        Args:
            name: Catalog name
            cascade: Drop all contents
            if_exists: Don't error if not exists

        Warning:
            CASCADE will delete all schemas and tables!
        """
        exists_clause = "IF EXISTS " if if_exists else ""
        cascade_clause = " CASCADE" if cascade else ""

        sql = f"DROP CATALOG {exists_clause}{name}{cascade_clause}"
        self.spark.sql(sql)
        logger.info(f"Dropped catalog: {name}")

    def list_catalogs(self) -> List[CatalogInfo]:
        """
        List all accessible catalogs.

        Returns:
            List of CatalogInfo objects
        """
        df = self.spark.sql("SHOW CATALOGS")
        return [
            CatalogInfo(name=row["catalog"])
            for row in df.collect()
        ]

    def get_catalog_info(self, name: str) -> CatalogInfo:
        """
        Get detailed information about a catalog.

        Args:
            name: Catalog name

        Returns:
            CatalogInfo with details
        """
        df = self.spark.sql(f"DESCRIBE CATALOG {name}")
        info_dict = {row["info_name"]: row["info_value"] for row in df.collect()}

        return CatalogInfo(
            name=name,
            comment=info_dict.get("Comment"),
            owner=info_dict.get("Owner"),
            created_at=info_dict.get("Created At")
        )

    # =========================================================================
    # SCHEMA OPERATIONS
    # =========================================================================

    def create_schema(
        self,
        catalog: str,
        schema: str,
        comment: Optional[str] = None,
        if_not_exists: bool = True
    ) -> None:
        """
        Create a new schema in a catalog.

        Args:
            catalog: Catalog name
            schema: Schema name
            comment: Optional description
            if_not_exists: Don't error if exists

        Example:
            >>> uc.create_schema("production", "bronze", comment="Raw ingestion layer")
        """
        exists_clause = "IF NOT EXISTS " if if_not_exists else ""
        comment_clause = f" COMMENT '{comment}'" if comment else ""

        full_name = f"{catalog}.{schema}"
        sql = f"CREATE SCHEMA {exists_clause}{full_name}{comment_clause}"
        self.spark.sql(sql)
        logger.info(f"Created schema: {full_name}")

    def drop_schema(
        self,
        catalog: str,
        schema: str,
        cascade: bool = False,
        if_exists: bool = True
    ) -> None:
        """
        Drop a schema.

        Args:
            catalog: Catalog name
            schema: Schema name
            cascade: Drop all contents
            if_exists: Don't error if not exists
        """
        exists_clause = "IF EXISTS " if if_exists else ""
        cascade_clause = " CASCADE" if cascade else ""

        full_name = f"{catalog}.{schema}"
        sql = f"DROP SCHEMA {exists_clause}{full_name}{cascade_clause}"
        self.spark.sql(sql)
        logger.info(f"Dropped schema: {full_name}")

    def list_schemas(self, catalog: str) -> List[SchemaInfo]:
        """
        List all schemas in a catalog.

        Args:
            catalog: Catalog name

        Returns:
            List of SchemaInfo objects
        """
        df = self.spark.sql(f"SHOW SCHEMAS IN {catalog}")
        return [
            SchemaInfo(
                name=row["databaseName"],
                catalog_name=catalog
            )
            for row in df.collect()
        ]

    # =========================================================================
    # TABLE OPERATIONS
    # =========================================================================

    def list_tables(
        self,
        catalog: str,
        schema: str
    ) -> List[TableInfo]:
        """
        List all tables in a schema.

        Args:
            catalog: Catalog name
            schema: Schema name

        Returns:
            List of TableInfo objects
        """
        full_schema = f"{catalog}.{schema}"
        df = self.spark.sql(f"SHOW TABLES IN {full_schema}")

        tables = []
        for row in df.collect():
            tables.append(TableInfo(
                name=row["tableName"],
                schema_name=schema,
                catalog_name=catalog,
                table_type="UNKNOWN"  # Would need DESCRIBE for details
            ))

        return tables

    def get_table_info(
        self,
        catalog: str,
        schema: str,
        table: str
    ) -> TableInfo:
        """
        Get detailed information about a table.

        Args:
            catalog: Catalog name
            schema: Schema name
            table: Table name

        Returns:
            TableInfo with details
        """
        full_name = f"{catalog}.{schema}.{table}"
        df = self.spark.sql(f"DESCRIBE TABLE EXTENDED {full_name}")

        info_dict = {}
        columns = []

        for row in df.collect():
            col_name = row["col_name"]
            data_type = row["data_type"]

            if col_name and not col_name.startswith("#"):
                if data_type:
                    columns.append({"name": col_name, "type": data_type})
                else:
                    info_dict[col_name] = row["comment"] or ""

        return TableInfo(
            name=table,
            schema_name=schema,
            catalog_name=catalog,
            table_type=info_dict.get("Type", "UNKNOWN"),
            data_source_format=info_dict.get("Provider"),
            location=info_dict.get("Location"),
            owner=info_dict.get("Owner"),
            columns=columns
        )

    def create_table(
        self,
        catalog: str,
        schema: str,
        table: str,
        columns: Dict[str, str],
        comment: Optional[str] = None,
        partition_by: Optional[List[str]] = None,
        table_properties: Optional[Dict[str, str]] = None,
        if_not_exists: bool = True
    ) -> None:
        """
        Create a managed Delta table.

        Args:
            catalog: Catalog name
            schema: Schema name
            table: Table name
            columns: Column name to type mapping
            comment: Table description
            partition_by: Partition columns
            table_properties: Delta table properties
            if_not_exists: Don't error if exists

        Example:
            >>> uc.create_table(
            ...     "production", "bronze", "events",
            ...     columns={"id": "BIGINT", "event_type": "STRING", "ts": "TIMESTAMP"},
            ...     partition_by=["event_type"],
            ...     comment="Raw event data"
            ... )
        """
        full_name = f"{catalog}.{schema}.{table}"
        exists_clause = "IF NOT EXISTS " if if_not_exists else ""

        # Build column definitions
        col_defs = ", ".join(f"{name} {dtype}" for name, dtype in columns.items())

        sql = f"CREATE TABLE {exists_clause}{full_name} ({col_defs}) USING DELTA"

        if partition_by:
            sql += f" PARTITIONED BY ({', '.join(partition_by)})"

        if comment:
            sql += f" COMMENT '{comment}'"

        if table_properties:
            props = ", ".join(f"'{k}'='{v}'" for k, v in table_properties.items())
            sql += f" TBLPROPERTIES ({props})"

        self.spark.sql(sql)
        logger.info(f"Created table: {full_name}")

    # =========================================================================
    # PERMISSION OPERATIONS
    # =========================================================================

    def grant_select(
        self,
        table_name: str,
        principal: str
    ) -> None:
        """
        Grant SELECT permission on a table.

        Args:
            table_name: Full table name (catalog.schema.table)
            principal: User, group, or service principal

        Example:
            >>> uc.grant_select("production.gold.sales_summary", "analysts_group")
        """
        sql = f"GRANT SELECT ON TABLE {table_name} TO `{principal}`"
        self.spark.sql(sql)
        logger.info(f"Granted SELECT on {table_name} to {principal}")

    def grant_all(
        self,
        table_name: str,
        principal: str
    ) -> None:
        """
        Grant all privileges on a table.

        Args:
            table_name: Full table name
            principal: User, group, or service principal
        """
        sql = f"GRANT ALL PRIVILEGES ON TABLE {table_name} TO `{principal}`"
        self.spark.sql(sql)
        logger.info(f"Granted ALL PRIVILEGES on {table_name} to {principal}")

    def revoke_select(
        self,
        table_name: str,
        principal: str
    ) -> None:
        """
        Revoke SELECT permission from a table.

        Args:
            table_name: Full table name
            principal: User, group, or service principal
        """
        sql = f"REVOKE SELECT ON TABLE {table_name} FROM `{principal}`"
        self.spark.sql(sql)
        logger.info(f"Revoked SELECT on {table_name} from {principal}")

    def show_grants(self, table_name: str) -> List[Dict[str, str]]:
        """
        Show all grants on a table.

        Args:
            table_name: Full table name

        Returns:
            List of grant information
        """
        df = self.spark.sql(f"SHOW GRANTS ON TABLE {table_name}")
        return [row.asDict() for row in df.collect()]

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def set_current_catalog(self, catalog: str) -> None:
        """
        Set the current catalog for subsequent operations.

        Args:
            catalog: Catalog name
        """
        self.spark.sql(f"USE CATALOG {catalog}")

    def set_current_schema(self, schema: str) -> None:
        """
        Set the current schema for subsequent operations.

        Args:
            schema: Schema name
        """
        self.spark.sql(f"USE SCHEMA {schema}")

    def get_table_lineage(self, table_name: str) -> Dict[str, Any]:
        """
        Get lineage information for a table.

        Note: Requires Databricks Runtime with lineage enabled.

        Args:
            table_name: Full table name

        Returns:
            Lineage information dictionary
        """
        # Lineage API is available via REST API in Databricks
        # This is a placeholder for the concept
        logger.warning("Table lineage requires Databricks REST API access")
        return {"table": table_name, "lineage": "not_implemented"}

    def add_table_tags(
        self,
        table_name: str,
        tags: Dict[str, str]
    ) -> None:
        """
        Add tags to a table for classification.

        Args:
            table_name: Full table name
            tags: Dictionary of tag key-value pairs

        Example:
            >>> uc.add_table_tags(
            ...     "prod.sales.customers",
            ...     {"pii": "true", "data_owner": "sales_team"}
            ... )
        """
        for key, value in tags.items():
            sql = f"ALTER TABLE {table_name} SET TAGS ('{key}' = '{value}')"
            self.spark.sql(sql)

        logger.info(f"Added tags to {table_name}: {tags}")
