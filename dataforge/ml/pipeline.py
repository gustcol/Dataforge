"""
DataForge ML Pipeline

Machine learning pipeline abstraction for organizing and executing
ML workflows across different engines.

Features:
    - Stage-based pipeline composition
    - Transform and model stages
    - Pipeline persistence
    - Execution tracking

Best Practices:
    1. Separate feature engineering from model training
    2. Use pipelines for reproducibility
    3. Version your pipelines
    4. Log pipeline parameters to MLflow

Example:
    >>> from dataforge.ml import MLPipeline, TransformStage, ModelStage
    >>>
    >>> # Build pipeline
    >>> pipeline = MLPipeline("customer_churn")
    >>> pipeline.add_stage(TransformStage("scale", StandardScaler(), ["amount"]))
    >>> pipeline.add_stage(TransformStage("encode", OneHotEncoder(), "category"))
    >>> pipeline.add_stage(ModelStage("train", LogisticRegression()))
    >>>
    >>> # Execute pipeline
    >>> result = pipeline.fit(train_df, target="churn")
    >>> predictions = pipeline.transform(test_df)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union
from datetime import datetime
import logging
import json

logger = logging.getLogger(__name__)


@dataclass
class StageResult:
    """Result from executing a pipeline stage.

    Attributes:
        stage_name: Name of the stage
        success: Whether stage completed successfully
        duration_seconds: Execution time
        metrics: Optional metrics from the stage
        error: Error message if failed
    """
    stage_name: str
    success: bool
    duration_seconds: float = 0.0
    metrics: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class PipelineResult:
    """Result from executing a complete pipeline.

    Attributes:
        pipeline_name: Name of the pipeline
        success: Whether all stages completed
        total_duration_seconds: Total execution time
        stage_results: Results from each stage
        output: Final output (DataFrame or model)
    """
    pipeline_name: str
    success: bool
    total_duration_seconds: float = 0.0
    stage_results: List[StageResult] = field(default_factory=list)
    output: Any = None


class PipelineStage(ABC):
    """
    Abstract base class for pipeline stages.

    All pipeline stages must implement fit() and transform() methods.
    """

    def __init__(self, name: str) -> None:
        """
        Initialize stage.

        Args:
            name: Stage name for identification
        """
        self.name = name
        self.is_fitted = False

    @abstractmethod
    def fit(self, df: Any, **kwargs: Any) -> "PipelineStage":
        """
        Fit the stage on training data.

        Args:
            df: Training DataFrame
            **kwargs: Additional arguments

        Returns:
            Self for method chaining
        """
        pass

    @abstractmethod
    def transform(self, df: Any) -> Any:
        """
        Transform data using fitted stage.

        Args:
            df: DataFrame to transform

        Returns:
            Transformed DataFrame
        """
        pass

    def fit_transform(self, df: Any, **kwargs: Any) -> Any:
        """Fit and transform in one step."""
        return self.fit(df, **kwargs).transform(df)


class TransformStage(PipelineStage):
    """
    Pipeline stage for data transformations.

    Wraps transformers (scalers, encoders, etc.) as pipeline stages.

    Example:
        >>> stage = TransformStage(
        ...     "scale_features",
        ...     StandardScaler(),
        ...     columns=["amount", "price"]
        ... )
        >>> pipeline.add_stage(stage)
    """

    def __init__(
        self,
        name: str,
        transformer: Any,
        columns: Optional[Union[str, List[str]]] = None
    ) -> None:
        """
        Initialize transform stage.

        Args:
            name: Stage name
            transformer: Transformer object with fit/transform methods
            columns: Column(s) to transform
        """
        super().__init__(name)
        self.transformer = transformer
        self.columns = columns if isinstance(columns, list) or columns is None else [columns]

    def fit(self, df: Any, **kwargs: Any) -> "TransformStage":
        """Fit transformer on data."""
        if hasattr(self.transformer, 'fit'):
            if self.columns:
                if len(self.columns) == 1:
                    self.transformer.fit(df, self.columns[0])
                else:
                    self.transformer.fit(df, self.columns)
            else:
                self.transformer.fit(df)

        self.is_fitted = True
        logger.info(f"Stage '{self.name}' fitted")
        return self

    def transform(self, df: Any) -> Any:
        """Apply transformation."""
        if hasattr(self.transformer, 'transform'):
            return self.transformer.transform(df)
        return df


class FunctionStage(PipelineStage):
    """
    Pipeline stage for custom functions.

    Wraps any callable as a pipeline stage.

    Example:
        >>> def clean_data(df):
        ...     return df.dropna()
        >>>
        >>> stage = FunctionStage("clean", clean_data)
        >>> pipeline.add_stage(stage)
    """

    def __init__(
        self,
        name: str,
        transform_func: Callable[[Any], Any],
        fit_func: Optional[Callable[[Any], None]] = None
    ) -> None:
        """
        Initialize function stage.

        Args:
            name: Stage name
            transform_func: Function to apply during transform
            fit_func: Optional function to call during fit
        """
        super().__init__(name)
        self.transform_func = transform_func
        self.fit_func = fit_func

    def fit(self, df: Any, **kwargs: Any) -> "FunctionStage":
        """Execute fit function if provided."""
        if self.fit_func:
            self.fit_func(df)
        self.is_fitted = True
        return self

    def transform(self, df: Any) -> Any:
        """Apply transform function."""
        return self.transform_func(df)


class ModelStage(PipelineStage):
    """
    Pipeline stage for ML models.

    Wraps ML models (sklearn, Spark MLlib, etc.) as pipeline stages.

    Example:
        >>> from sklearn.linear_model import LogisticRegression
        >>>
        >>> stage = ModelStage(
        ...     "classifier",
        ...     LogisticRegression(),
        ...     feature_columns=["f1", "f2"],
        ...     target_column="label"
        ... )
        >>> pipeline.add_stage(stage)
    """

    def __init__(
        self,
        name: str,
        model: Any,
        feature_columns: Optional[List[str]] = None,
        target_column: Optional[str] = None
    ) -> None:
        """
        Initialize model stage.

        Args:
            name: Stage name
            model: ML model with fit/predict methods
            feature_columns: Feature column names
            target_column: Target column name
        """
        super().__init__(name)
        self.model = model
        self.feature_columns = feature_columns
        self.target_column = target_column
        self.metrics: Dict[str, Any] = {}

    def fit(self, df: Any, **kwargs: Any) -> "ModelStage":
        """Fit model on training data."""
        target = kwargs.get("target", self.target_column)
        features = kwargs.get("features", self.feature_columns)

        if hasattr(df, 'toPandas'):
            # Spark DataFrame - use Spark MLlib or convert
            pdf = df.toPandas()
            X = pdf[features] if features else pdf.drop(columns=[target])
            y = pdf[target]
            self.model.fit(X, y)
        else:
            # Pandas/cuDF
            X = df[features] if features else df.drop(columns=[target])
            y = df[target]
            self.model.fit(X, y)

        self.is_fitted = True
        logger.info(f"Model stage '{self.name}' fitted")
        return self

    def transform(self, df: Any) -> Any:
        """Generate predictions."""
        if not self.is_fitted:
            raise ValueError(f"Model stage '{self.name}' not fitted")

        features = self.feature_columns
        prediction_col = f"{self.name}_prediction"

        if hasattr(df, 'toPandas'):
            from pyspark.sql import functions as F
            from pyspark.sql.types import DoubleType

            pdf = df.toPandas()
            X = pdf[features] if features else pdf
            predictions = self.model.predict(X)

            # Add predictions back
            pdf[prediction_col] = predictions

            # Convert back to Spark (simplified)
            spark = df.sparkSession
            return spark.createDataFrame(pdf)
        else:
            X = df[features] if features else df
            df[prediction_col] = self.model.predict(X)
            return df

    def predict(self, df: Any) -> Any:
        """Alias for transform."""
        return self.transform(df)

    def evaluate(self, df: Any, target: str) -> Dict[str, float]:
        """
        Evaluate model on data.

        Args:
            df: DataFrame with features and target
            target: Target column name

        Returns:
            Dictionary of metrics
        """
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        if hasattr(df, 'toPandas'):
            pdf = df.toPandas()
        else:
            pdf = df

        features = self.feature_columns
        X = pdf[features] if features else pdf.drop(columns=[target])
        y_true = pdf[target]
        y_pred = self.model.predict(X)

        self.metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, average='weighted', zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
        }

        logger.info(f"Model evaluation: {self.metrics}")
        return self.metrics


class MLPipeline:
    """
    Machine learning pipeline for organizing and executing ML workflows.

    Provides a structured way to chain transformations and models,
    with support for fitting, transforming, and persistence.

    Example:
        >>> from dataforge.ml import MLPipeline, TransformStage, ModelStage
        >>> from sklearn.ensemble import RandomForestClassifier
        >>>
        >>> # Create pipeline
        >>> pipeline = MLPipeline("churn_prediction")
        >>>
        >>> # Add preprocessing stages
        >>> pipeline.add_stage(TransformStage("fill_na", FillNaTransformer()))
        >>> pipeline.add_stage(TransformStage("scale", StandardScaler(), ["amount"]))
        >>> pipeline.add_stage(TransformStage("encode", OneHotEncoder(), "category"))
        >>>
        >>> # Add model stage
        >>> pipeline.add_stage(ModelStage(
        ...     "classifier",
        ...     RandomForestClassifier(n_estimators=100),
        ...     feature_columns=["amount_scaled", "category_A", "category_B"],
        ...     target_column="churn"
        ... ))
        >>>
        >>> # Fit pipeline
        >>> result = pipeline.fit(train_df)
        >>>
        >>> # Transform/predict
        >>> predictions = pipeline.transform(test_df)
        >>>
        >>> # Evaluate
        >>> metrics = pipeline.evaluate(test_df, target="churn")
    """

    def __init__(self, name: str, description: str = "") -> None:
        """
        Initialize pipeline.

        Args:
            name: Pipeline name
            description: Optional description
        """
        self.name = name
        self.description = description
        self.stages: List[PipelineStage] = []
        self.is_fitted = False
        self.created_at = datetime.now()
        self.fitted_at: Optional[datetime] = None
        self.metadata: Dict[str, Any] = {}

    def add_stage(self, stage: PipelineStage) -> "MLPipeline":
        """
        Add a stage to the pipeline.

        Args:
            stage: Pipeline stage to add

        Returns:
            Self for method chaining
        """
        self.stages.append(stage)
        logger.debug(f"Added stage '{stage.name}' to pipeline '{self.name}'")
        return self

    def fit(self, df: Any, **kwargs: Any) -> PipelineResult:
        """
        Fit all stages on training data.

        Args:
            df: Training DataFrame
            **kwargs: Additional arguments (e.g., target column)

        Returns:
            PipelineResult with execution details
        """
        import time

        result = PipelineResult(
            pipeline_name=self.name,
            success=True
        )

        start_time = time.time()
        current_df = df

        for stage in self.stages:
            stage_start = time.time()

            try:
                stage.fit(current_df, **kwargs)
                current_df = stage.transform(current_df)

                stage_result = StageResult(
                    stage_name=stage.name,
                    success=True,
                    duration_seconds=time.time() - stage_start
                )

            except Exception as e:
                stage_result = StageResult(
                    stage_name=stage.name,
                    success=False,
                    duration_seconds=time.time() - stage_start,
                    error=str(e)
                )
                result.success = False
                result.stage_results.append(stage_result)
                logger.error(f"Pipeline stage '{stage.name}' failed: {e}")
                break

            result.stage_results.append(stage_result)

        result.total_duration_seconds = time.time() - start_time
        result.output = current_df

        if result.success:
            self.is_fitted = True
            self.fitted_at = datetime.now()
            logger.info(
                f"Pipeline '{self.name}' fitted successfully "
                f"in {result.total_duration_seconds:.2f}s"
            )

        return result

    def transform(self, df: Any) -> Any:
        """
        Apply all fitted stages to data.

        Args:
            df: DataFrame to transform

        Returns:
            Transformed DataFrame
        """
        if not self.is_fitted:
            raise ValueError(f"Pipeline '{self.name}' not fitted. Call fit() first.")

        current_df = df

        for stage in self.stages:
            current_df = stage.transform(current_df)

        return current_df

    def fit_transform(self, df: Any, **kwargs: Any) -> Any:
        """Fit and transform in one step."""
        result = self.fit(df, **kwargs)
        return result.output

    def evaluate(self, df: Any, target: str) -> Dict[str, float]:
        """
        Evaluate pipeline on data.

        Finds the model stage and evaluates it.

        Args:
            df: DataFrame with features and target
            target: Target column name

        Returns:
            Dictionary of evaluation metrics
        """
        # Transform data first
        transformed = self.transform(df)

        # Find model stage
        for stage in reversed(self.stages):
            if isinstance(stage, ModelStage):
                return stage.evaluate(transformed, target)

        raise ValueError("No model stage found in pipeline")

    def get_stage(self, name: str) -> Optional[PipelineStage]:
        """Get a stage by name."""
        for stage in self.stages:
            if stage.name == name:
                return stage
        return None

    def remove_stage(self, name: str) -> bool:
        """Remove a stage by name."""
        for i, stage in enumerate(self.stages):
            if stage.name == name:
                self.stages.pop(i)
                return True
        return False

    def get_params(self) -> Dict[str, Any]:
        """Get all pipeline parameters."""
        params = {
            "pipeline_name": self.name,
            "description": self.description,
            "num_stages": len(self.stages),
            "stages": [],
        }

        for stage in self.stages:
            stage_params = {
                "name": stage.name,
                "type": type(stage).__name__,
                "is_fitted": stage.is_fitted,
            }

            # Extract model/transformer params if available
            if hasattr(stage, 'model') and hasattr(stage.model, 'get_params'):
                stage_params["model_params"] = stage.model.get_params()
            elif hasattr(stage, 'transformer') and hasattr(stage.transformer, '__dict__'):
                stage_params["transformer_params"] = {
                    k: v for k, v in stage.transformer.__dict__.items()
                    if not k.startswith('_')
                }

            params["stages"].append(stage_params)

        return params

    def summary(self) -> str:
        """Get human-readable pipeline summary."""
        lines = [
            "=" * 60,
            f"Pipeline: {self.name}",
            "=" * 60,
            f"Description: {self.description or 'N/A'}",
            f"Stages: {len(self.stages)}",
            f"Fitted: {self.is_fitted}",
            f"Created: {self.created_at}",
            f"Fitted at: {self.fitted_at or 'N/A'}",
            "",
            "Stages:",
            "-" * 40,
        ]

        for i, stage in enumerate(self.stages):
            fitted_mark = "✓" if stage.is_fitted else "○"
            lines.append(f"  {i+1}. [{fitted_mark}] {stage.name} ({type(stage).__name__})")

        lines.append("=" * 60)
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert pipeline metadata to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "num_stages": len(self.stages),
            "stage_names": [s.name for s in self.stages],
            "is_fitted": self.is_fitted,
            "created_at": str(self.created_at),
            "fitted_at": str(self.fitted_at) if self.fitted_at else None,
            "metadata": self.metadata,
        }

    def __repr__(self) -> str:
        return f"MLPipeline(name='{self.name}', stages={len(self.stages)}, fitted={self.is_fitted})"
