"""
DataForge Databricks Module

Comprehensive Databricks integration including:
    - Delta Lake operations (OPTIMIZE, VACUUM, ZORDER)
    - Unity Catalog management (catalogs, schemas, tables)
    - Photon optimization utilities
    - Databricks context management
    - Cluster configuration recommendations

Unity Catalog Integration:
    Unity Catalog provides a unified governance solution for all data assets.
    This module supports the three-level namespace: catalog.schema.table

Delta Lake Features:
    - ACID transactions
    - Time travel (versioning)
    - Schema evolution
    - Change data capture
    - OPTIMIZE and VACUUM operations

Photon Engine:
    - Vectorized query execution
    - 2-8x performance improvement
    - Native C++ engine
    - Delta Lake optimized

Example:
    >>> from dataforge.databricks import (
    ...     DeltaTableManager,
    ...     UnityCatalogManager,
    ...     DatabricksContext,
    ...     PhotonAnalyzer
    ... )
    >>>
    >>> # Delta Lake operations
    >>> delta = DeltaTableManager(spark)
    >>> delta.optimize("my_table", z_order_by=["date", "region"])
    >>> delta.vacuum("my_table", retention_hours=168)
    >>>
    >>> # Unity Catalog operations
    >>> uc = UnityCatalogManager(spark)
    >>> tables = uc.list_tables("my_catalog", "my_schema")
    >>> uc.grant_select("my_catalog.my_schema.my_table", "analysts_group")
    >>>
    >>> # Context management
    >>> ctx = DatabricksContext()
    >>> if ctx.is_databricks:
    ...     secret = ctx.get_secret("scope", "key")
    >>>
    >>> # Photon analysis
    >>> analyzer = PhotonAnalyzer(spark)
    >>> report = analyzer.analyze_query("SELECT * FROM sales")
"""

from dataforge.databricks.delta import (
    DeltaTableManager,
    DeltaTableInfo,
    DeltaHistory,
)
from dataforge.databricks.unity_catalog import (
    UnityCatalogManager,
    CatalogInfo,
    SchemaInfo,
    TableInfo,
)
from dataforge.databricks.context import (
    DatabricksContext,
    ClusterInfo,
    RuntimeInfo,
    get_context,
)
from dataforge.databricks.optimizations import (
    OptimizationConfig,
    optimize_spark_config,
    get_photon_recommendations,
    configure_for_large_shuffle,
    configure_for_streaming,
    configure_for_ml,
    get_cluster_recommendations,
    analyze_query_performance,
    enable_query_watchdog,
)
from dataforge.databricks.photon import (
    PhotonAnalyzer,
    PhotonCompatibility,
    check_photon_compatibility,
    configure_photon_optimal,
)

__all__ = [
    # Delta Lake
    "DeltaTableManager",
    "DeltaTableInfo",
    "DeltaHistory",
    # Unity Catalog
    "UnityCatalogManager",
    "CatalogInfo",
    "SchemaInfo",
    "TableInfo",
    # Context
    "DatabricksContext",
    "ClusterInfo",
    "RuntimeInfo",
    "get_context",
    # Optimizations
    "OptimizationConfig",
    "optimize_spark_config",
    "get_photon_recommendations",
    "configure_for_large_shuffle",
    "configure_for_streaming",
    "configure_for_ml",
    "get_cluster_recommendations",
    "analyze_query_performance",
    "enable_query_watchdog",
    # Photon
    "PhotonAnalyzer",
    "PhotonCompatibility",
    "check_photon_compatibility",
    "configure_photon_optimal",
]
