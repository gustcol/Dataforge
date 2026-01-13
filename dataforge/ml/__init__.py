"""
DataForge ML Module

Machine learning utilities and integrations for the DataForge framework.

Features:
    - Feature engineering utilities
    - ML pipeline abstraction
    - MLflow integration
    - Cross-engine ML support

Example:
    >>> from dataforge.ml import FeatureEngineer, MLPipeline, MLflowTracker
    >>>
    >>> # Feature engineering
    >>> fe = FeatureEngineer()
    >>> df = fe.one_hot_encode(df, "category")
    >>> df = fe.standard_scale(df, ["amount", "quantity"])
    >>>
    >>> # MLflow tracking
    >>> with MLflowTracker("my-experiment") as tracker:
    ...     tracker.log_params({"learning_rate": 0.01})
    ...     # ... train model ...
    ...     tracker.log_metrics({"accuracy": 0.95})
"""

from dataforge.ml.features import (
    FeatureEngineer,
    OneHotEncoder,
    LabelEncoder,
    StandardScaler,
    MinMaxScaler,
    Binner,
)
from dataforge.ml.pipeline import (
    MLPipeline,
    PipelineStage,
    TransformStage,
    ModelStage,
)
from dataforge.ml.mlflow_utils import (
    MLflowTracker,
    log_dataframe_info,
    log_model_metrics,
    load_model,
)

__all__ = [
    # Features
    "FeatureEngineer",
    "OneHotEncoder",
    "LabelEncoder",
    "StandardScaler",
    "MinMaxScaler",
    "Binner",
    # Pipeline
    "MLPipeline",
    "PipelineStage",
    "TransformStage",
    "ModelStage",
    # MLflow
    "MLflowTracker",
    "log_dataframe_info",
    "log_model_metrics",
    "load_model",
]
