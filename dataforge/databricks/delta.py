"""
DataForge Delta Lake Manager

Delta Lake is an open-source storage layer that brings ACID transactions
to Apache Spark and big data workloads.

Features:
    - ACID transactions
    - Scalable metadata handling
    - Time travel (data versioning)
    - Unified batch and streaming
    - Schema enforcement and evolution
    - Audit history

Optimization Operations:
    - OPTIMIZE: Compact small files
    - VACUUM: Remove old files
    - ZORDER: Co-locate related data
    - Auto-compaction: Automatic file compaction

Best Practices:
    1. Run OPTIMIZE regularly (daily for write-heavy tables)
    2. VACUUM after 7+ days retention
    3. Use ZORDER on frequently filtered columns
    4. Enable auto-compaction for streaming tables
    5. Monitor file sizes (target: 128MB - 1GB)

Example:
    >>> from dataforge.databricks import DeltaTableManager
    >>>
    >>> delta = DeltaTableManager(spark)
    >>>
    >>> # Optimize table with Z-ordering
    >>> delta.optimize("catalog.schema.table", z_order_by=["date", "region"])
    >>>
    >>> # Vacuum old files
    >>> delta.vacuum("catalog.schema.table", retention_hours=168)
    >>>
    >>> # Time travel query
    >>> df = delta.read_version("catalog.schema.table", version=10)
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, TYPE_CHECKING
import logging

if TYPE_CHECKING:
    from pyspark.sql import SparkSession, DataFrame

logger = logging.getLogger(__name__)


@dataclass
class DeltaTableInfo:
    """Information about a Delta table."""
    name: str
    location: str
    num_files: int
    size_bytes: int
    num_partitions: int
    created_at: Optional[str] = None
    last_modified: Optional[str] = None
    properties: Optional[Dict[str, str]] = None


@dataclass
class DeltaHistory:
    """History entry for a Delta table."""
    version: int
    timestamp: datetime
    operation: str
    user_name: Optional[str] = None
    operation_parameters: Optional[Dict[str, str]] = None
    metrics: Optional[Dict[str, Any]] = None


class DeltaTableManager:
    """
    Manager for Delta Lake table operations.

    Provides utilities for optimizing, maintaining, and querying
    Delta Lake tables.

    Example:
        >>> delta = DeltaTableManager(spark)
        >>>
        >>> # Basic optimization
        >>> delta.optimize("my_catalog.my_schema.my_table")
        >>>
        >>> # With Z-ordering for better query performance
        >>> delta.optimize(
        ...     "my_catalog.my_schema.my_table",
        ...     z_order_by=["customer_id", "date"]
        ... )
        >>>
        >>> # Vacuum old files
        >>> delta.vacuum("my_catalog.my_schema.my_table")
    """

    def __init__(self, spark: "SparkSession") -> None:
        """
        Initialize Delta Table Manager.

        Args:
            spark: SparkSession
        """
        self.spark = spark

    # =========================================================================
    # OPTIMIZATION OPERATIONS
    # =========================================================================

    def optimize(
        self,
        table_name: str,
        z_order_by: Optional[List[str]] = None,
        where: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Optimize Delta table by compacting small files.

        OPTIMIZE consolidates many small files into fewer larger files.
        This improves read performance significantly.

        Args:
            table_name: Full table name (catalog.schema.table)
            z_order_by: Columns for Z-ordering (improves query performance)
            where: Partition filter to optimize specific partitions

        Returns:
            Optimization metrics

        Best Practice:
            - Run during off-peak hours
            - For large tables, use WHERE to optimize incrementally
            - Z-order on columns frequently used in WHERE clauses

        Example:
            >>> # Basic optimize
            >>> delta.optimize("prod.sales.orders")
            >>>
            >>> # With Z-ordering
            >>> delta.optimize(
            ...     "prod.sales.orders",
            ...     z_order_by=["customer_id", "order_date"]
            ... )
            >>>
            >>> # Optimize specific partition
            >>> delta.optimize(
            ...     "prod.sales.orders",
            ...     where="order_date >= '2024-01-01'"
            ... )
        """
        sql = f"OPTIMIZE {table_name}"

        if where:
            sql += f" WHERE {where}"

        if z_order_by:
            z_order_cols = ", ".join(z_order_by)
            sql += f" ZORDER BY ({z_order_cols})"

        logger.info(f"Running OPTIMIZE on {table_name}")
        result = self.spark.sql(sql)

        # Collect metrics
        metrics = {}
        try:
            row = result.collect()[0]
            metrics = row.asDict()
        except Exception:
            pass

        logger.info(f"OPTIMIZE completed for {table_name}")
        return metrics

    def vacuum(
        self,
        table_name: str,
        retention_hours: int = 168  # 7 days
    ) -> None:
        """
        Remove old files from Delta table.

        VACUUM removes files no longer referenced by the Delta table
        that are older than the retention period.

        Args:
            table_name: Full table name
            retention_hours: Minimum file age to delete (default: 168 hours / 7 days)

        Warning:
            - Never set retention < 168 hours in production
            - Time travel won't work for removed versions
            - Concurrent readers may fail if retention too low

        Example:
            >>> # Standard 7-day retention
            >>> delta.vacuum("prod.sales.orders")
            >>>
            >>> # 30-day retention
            >>> delta.vacuum("prod.sales.orders", retention_hours=720)
        """
        if retention_hours < 168:
            logger.warning(
                f"Retention {retention_hours}h is below recommended 168h (7 days). "
                "This may affect concurrent readers."
            )

        sql = f"VACUUM {table_name} RETAIN {retention_hours} HOURS"
        logger.info(f"Running VACUUM on {table_name} with {retention_hours}h retention")

        self.spark.sql(sql)
        logger.info(f"VACUUM completed for {table_name}")

    def auto_compact(
        self,
        table_name: str,
        enabled: bool = True
    ) -> None:
        """
        Enable or disable auto-compaction for a table.

        Auto-compaction automatically runs OPTIMIZE after writes.

        Args:
            table_name: Full table name
            enabled: Enable or disable

        Best Practice:
            Enable for streaming tables to prevent small file accumulation.
        """
        value = "true" if enabled else "false"
        sql = f"""
            ALTER TABLE {table_name}
            SET TBLPROPERTIES ('delta.autoOptimize.autoCompact' = '{value}')
        """
        self.spark.sql(sql)
        logger.info(f"Auto-compact {'enabled' if enabled else 'disabled'} for {table_name}")

    def optimize_write(
        self,
        table_name: str,
        enabled: bool = True
    ) -> None:
        """
        Enable or disable optimized writes for a table.

        Optimized writes reduce the number of files during writes.

        Args:
            table_name: Full table name
            enabled: Enable or disable
        """
        value = "true" if enabled else "false"
        sql = f"""
            ALTER TABLE {table_name}
            SET TBLPROPERTIES ('delta.autoOptimize.optimizeWrite' = '{value}')
        """
        self.spark.sql(sql)
        logger.info(f"Optimize write {'enabled' if enabled else 'disabled'} for {table_name}")

    # =========================================================================
    # TIME TRAVEL OPERATIONS
    # =========================================================================

    def read_version(
        self,
        table_name: str,
        version: int
    ) -> "DataFrame":
        """
        Read a specific version of a Delta table.

        Args:
            table_name: Full table name
            version: Version number to read

        Returns:
            DataFrame at specified version

        Example:
            >>> # Read version 10
            >>> df = delta.read_version("prod.sales.orders", version=10)
        """
        return self.spark.read.format("delta").option(
            "versionAsOf", version
        ).table(table_name)

    def read_timestamp(
        self,
        table_name: str,
        timestamp: str
    ) -> "DataFrame":
        """
        Read Delta table as of a specific timestamp.

        Args:
            table_name: Full table name
            timestamp: Timestamp string (e.g., "2024-01-01 00:00:00")

        Returns:
            DataFrame at specified timestamp

        Example:
            >>> df = delta.read_timestamp(
            ...     "prod.sales.orders",
            ...     "2024-01-01 00:00:00"
            ... )
        """
        return self.spark.read.format("delta").option(
            "timestampAsOf", timestamp
        ).table(table_name)

    def get_history(
        self,
        table_name: str,
        limit: Optional[int] = None
    ) -> List[DeltaHistory]:
        """
        Get version history of a Delta table.

        Args:
            table_name: Full table name
            limit: Maximum number of history entries

        Returns:
            List of DeltaHistory entries

        Example:
            >>> history = delta.get_history("prod.sales.orders", limit=10)
            >>> for h in history:
            ...     print(f"v{h.version}: {h.operation} at {h.timestamp}")
        """
        sql = f"DESCRIBE HISTORY {table_name}"
        if limit:
            sql += f" LIMIT {limit}"

        df = self.spark.sql(sql)

        history = []
        for row in df.collect():
            history.append(DeltaHistory(
                version=row["version"],
                timestamp=row["timestamp"],
                operation=row["operation"],
                user_name=row.get("userName"),
                operation_parameters=row.get("operationParameters"),
                metrics=row.get("operationMetrics")
            ))

        return history

    def restore(
        self,
        table_name: str,
        version: Optional[int] = None,
        timestamp: Optional[str] = None
    ) -> None:
        """
        Restore Delta table to a previous version.

        Args:
            table_name: Full table name
            version: Version to restore to
            timestamp: Timestamp to restore to

        Example:
            >>> delta.restore("prod.sales.orders", version=5)
        """
        if version is not None:
            sql = f"RESTORE TABLE {table_name} TO VERSION AS OF {version}"
        elif timestamp is not None:
            sql = f"RESTORE TABLE {table_name} TO TIMESTAMP AS OF '{timestamp}'"
        else:
            raise ValueError("Either version or timestamp must be specified")

        logger.info(f"Restoring {table_name} to {'v' + str(version) if version else timestamp}")
        self.spark.sql(sql)
        logger.info(f"Restore completed for {table_name}")

    # =========================================================================
    # MERGE OPERATIONS
    # =========================================================================

    def merge(
        self,
        target_table: str,
        source_df: "DataFrame",
        merge_condition: str,
        when_matched_update: Optional[Dict[str, str]] = None,
        when_matched_delete: Optional[str] = None,
        when_not_matched_insert: Optional[Dict[str, str]] = None
    ) -> Dict[str, int]:
        """
        Perform MERGE (upsert) operation.

        MERGE allows atomic INSERT, UPDATE, and DELETE in a single operation.

        Args:
            target_table: Target Delta table
            source_df: Source DataFrame
            merge_condition: Join condition
            when_matched_update: Column updates when matched
            when_matched_delete: Delete condition when matched
            when_not_matched_insert: Insert columns when not matched

        Returns:
            Metrics dictionary with row counts

        Example:
            >>> delta.merge(
            ...     target_table="prod.sales.customers",
            ...     source_df=updates_df,
            ...     merge_condition="target.id = source.id",
            ...     when_matched_update={
            ...         "name": "source.name",
            ...         "email": "source.email",
            ...         "updated_at": "current_timestamp()"
            ...     },
            ...     when_not_matched_insert={
            ...         "id": "source.id",
            ...         "name": "source.name",
            ...         "email": "source.email",
            ...         "created_at": "current_timestamp()"
            ...     }
            ... )
        """
        from delta.tables import DeltaTable

        # Get Delta table
        target = DeltaTable.forName(self.spark, target_table)

        # Build merge
        merge_builder = target.alias("target").merge(
            source_df.alias("source"),
            merge_condition
        )

        # When matched - update
        if when_matched_update:
            merge_builder = merge_builder.whenMatchedUpdate(
                set=when_matched_update
            )

        # When matched - delete
        if when_matched_delete:
            merge_builder = merge_builder.whenMatchedDelete(
                condition=when_matched_delete
            )

        # When not matched - insert
        if when_not_matched_insert:
            merge_builder = merge_builder.whenNotMatchedInsert(
                values=when_not_matched_insert
            )

        # Execute
        logger.info(f"Executing MERGE on {target_table}")
        merge_builder.execute()

        return {"rows_affected": 0}  # Metrics not available from basic merge

    # =========================================================================
    # TABLE INFORMATION
    # =========================================================================

    def get_table_info(self, table_name: str) -> DeltaTableInfo:
        """
        Get detailed information about a Delta table.

        Args:
            table_name: Full table name

        Returns:
            DeltaTableInfo with table details
        """
        df = self.spark.sql(f"DESCRIBE DETAIL {table_name}")
        row = df.collect()[0]

        return DeltaTableInfo(
            name=table_name,
            location=row["location"],
            num_files=row["numFiles"],
            size_bytes=row["sizeInBytes"],
            num_partitions=len(row["partitionColumns"]) if row["partitionColumns"] else 0,
            properties=row.get("properties")
        )

    def get_file_stats(self, table_name: str) -> Dict[str, Any]:
        """
        Get file statistics for a Delta table.

        Args:
            table_name: Full table name

        Returns:
            Dictionary with file statistics
        """
        info = self.get_table_info(table_name)

        avg_file_size = info.size_bytes / info.num_files if info.num_files > 0 else 0

        return {
            "num_files": info.num_files,
            "total_size_gb": info.size_bytes / (1024 ** 3),
            "avg_file_size_mb": avg_file_size / (1024 ** 2),
            "needs_optimization": avg_file_size < 64 * 1024 * 1024,  # < 64MB
            "recommendation": (
                "Run OPTIMIZE" if avg_file_size < 64 * 1024 * 1024
                else "Table is well optimized"
            )
        }

    # =========================================================================
    # SCHEMA OPERATIONS
    # =========================================================================

    def add_column(
        self,
        table_name: str,
        column_name: str,
        data_type: str,
        comment: Optional[str] = None
    ) -> None:
        """
        Add a column to a Delta table.

        Args:
            table_name: Full table name
            column_name: New column name
            data_type: Spark SQL data type
            comment: Optional column comment

        Example:
            >>> delta.add_column(
            ...     "prod.sales.orders",
            ...     "discount_pct",
            ...     "DOUBLE",
            ...     comment="Discount percentage applied"
            ... )
        """
        comment_clause = f" COMMENT '{comment}'" if comment else ""
        sql = f"ALTER TABLE {table_name} ADD COLUMN {column_name} {data_type}{comment_clause}"
        self.spark.sql(sql)
        logger.info(f"Added column {column_name} to {table_name}")

    def change_column(
        self,
        table_name: str,
        column_name: str,
        new_name: Optional[str] = None,
        new_type: Optional[str] = None,
        new_comment: Optional[str] = None
    ) -> None:
        """
        Change a column in a Delta table.

        Args:
            table_name: Full table name
            column_name: Current column name
            new_name: New column name
            new_type: New data type
            new_comment: New comment
        """
        if new_name:
            sql = f"ALTER TABLE {table_name} RENAME COLUMN {column_name} TO {new_name}"
            self.spark.sql(sql)
            column_name = new_name

        if new_type:
            sql = f"ALTER TABLE {table_name} ALTER COLUMN {column_name} TYPE {new_type}"
            self.spark.sql(sql)

        if new_comment:
            sql = f"ALTER TABLE {table_name} ALTER COLUMN {column_name} COMMENT '{new_comment}'"
            self.spark.sql(sql)

        logger.info(f"Changed column {column_name} in {table_name}")

    def enable_change_data_feed(self, table_name: str) -> None:
        """
        Enable Change Data Feed for a table.

        CDF tracks row-level changes for incremental processing.

        Args:
            table_name: Full table name
        """
        sql = f"""
            ALTER TABLE {table_name}
            SET TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true')
        """
        self.spark.sql(sql)
        logger.info(f"Enabled Change Data Feed for {table_name}")

    def read_changes(
        self,
        table_name: str,
        starting_version: int,
        ending_version: Optional[int] = None
    ) -> "DataFrame":
        """
        Read changes from Change Data Feed.

        Args:
            table_name: Full table name
            starting_version: Start version (inclusive)
            ending_version: End version (inclusive, optional)

        Returns:
            DataFrame with changes

        Example:
            >>> changes = delta.read_changes(
            ...     "prod.sales.orders",
            ...     starting_version=100,
            ...     ending_version=110
            ... )
        """
        reader = self.spark.read.format("delta").option(
            "readChangeFeed", "true"
        ).option("startingVersion", starting_version)

        if ending_version:
            reader = reader.option("endingVersion", ending_version)

        return reader.table(table_name)
