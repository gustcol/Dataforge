"""
DataForge MLflow Integration

Utilities for integrating with MLflow for experiment tracking,
model versioning, and artifact management.

Features:
    - Experiment tracking
    - Parameter and metric logging
    - Model registration
    - Artifact management
    - Auto-logging support

Best Practices:
    1. Use consistent experiment naming
    2. Log all hyperparameters
    3. Track data versions/hashes
    4. Register production models
    5. Use model signatures

Example:
    >>> from dataforge.ml import MLflowTracker
    >>>
    >>> with MLflowTracker("my_experiment") as tracker:
    ...     tracker.log_params({"learning_rate": 0.01, "epochs": 100})
    ...     # ... train model ...
    ...     tracker.log_metrics({"accuracy": 0.95, "f1": 0.93})
    ...     tracker.log_model(model, "classifier")
"""

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
import logging
import os

logger = logging.getLogger(__name__)


@dataclass
class RunInfo:
    """Information about an MLflow run.

    Attributes:
        run_id: Unique run identifier
        experiment_id: Parent experiment ID
        status: Run status
        start_time: Run start timestamp
        end_time: Run end timestamp
        artifact_uri: URI for artifacts
    """
    run_id: str
    experiment_id: str
    status: str = "RUNNING"
    start_time: Optional[int] = None
    end_time: Optional[int] = None
    artifact_uri: Optional[str] = None


class MLflowTracker:
    """
    MLflow tracking context manager.

    Provides a convenient interface for tracking ML experiments,
    logging parameters, metrics, and models.

    Example:
        >>> with MLflowTracker("churn_prediction") as tracker:
        ...     # Log parameters
        ...     tracker.log_params({
        ...         "model_type": "RandomForest",
        ...         "n_estimators": 100,
        ...         "max_depth": 10
        ...     })
        ...
        ...     # Train model
        ...     model.fit(X_train, y_train)
        ...
        ...     # Log metrics
        ...     tracker.log_metrics({
        ...         "train_accuracy": train_acc,
        ...         "val_accuracy": val_acc
        ...     })
        ...
        ...     # Log model
        ...     tracker.log_model(model, "random_forest")
        ...
        >>> # Run is automatically ended when exiting context
    """

    def __init__(
        self,
        experiment_name: str,
        run_name: Optional[str] = None,
        tracking_uri: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Initialize MLflow tracker.

        Args:
            experiment_name: Name of the experiment
            run_name: Optional name for this run
            tracking_uri: MLflow tracking server URI
            tags: Optional tags for the run
        """
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.tracking_uri = tracking_uri
        self.tags = tags or {}
        self._run = None
        self._mlflow = None

    def __enter__(self) -> "MLflowTracker":
        """Start MLflow run."""
        self._setup_mlflow()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """End MLflow run."""
        if self._run is not None:
            self._mlflow.end_run()
            logger.info(f"MLflow run ended: {self._run.info.run_id}")

    def _setup_mlflow(self) -> None:
        """Setup MLflow tracking."""
        try:
            import mlflow
            self._mlflow = mlflow

            # Set tracking URI if provided
            if self.tracking_uri:
                mlflow.set_tracking_uri(self.tracking_uri)

            # Set or create experiment
            mlflow.set_experiment(self.experiment_name)

            # Start run
            self._run = mlflow.start_run(run_name=self.run_name)

            # Set tags
            for key, value in self.tags.items():
                mlflow.set_tag(key, value)

            logger.info(
                f"MLflow run started: {self._run.info.run_id} "
                f"in experiment '{self.experiment_name}'"
            )

        except ImportError:
            logger.warning("MLflow not installed. Install with: pip install mlflow")
            raise ImportError("MLflow is required for tracking. Install with: pip install mlflow")

    @property
    def run_id(self) -> Optional[str]:
        """Get current run ID."""
        return self._run.info.run_id if self._run else None

    @property
    def experiment_id(self) -> Optional[str]:
        """Get current experiment ID."""
        return self._run.info.experiment_id if self._run else None

    def log_param(self, key: str, value: Any) -> None:
        """
        Log a single parameter.

        Args:
            key: Parameter name
            value: Parameter value
        """
        if self._mlflow:
            self._mlflow.log_param(key, value)

    def log_params(self, params: Dict[str, Any]) -> None:
        """
        Log multiple parameters.

        Args:
            params: Dictionary of parameter names and values
        """
        if self._mlflow:
            self._mlflow.log_params(params)
            logger.debug(f"Logged {len(params)} parameters")

    def log_metric(self, key: str, value: float, step: Optional[int] = None) -> None:
        """
        Log a single metric.

        Args:
            key: Metric name
            value: Metric value
            step: Optional step number for time series metrics
        """
        if self._mlflow:
            self._mlflow.log_metric(key, value, step=step)

    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None
    ) -> None:
        """
        Log multiple metrics.

        Args:
            metrics: Dictionary of metric names and values
            step: Optional step number
        """
        if self._mlflow:
            self._mlflow.log_metrics(metrics, step=step)
            logger.debug(f"Logged {len(metrics)} metrics")

    def log_model(
        self,
        model: Any,
        artifact_path: str,
        registered_model_name: Optional[str] = None,
        signature: Optional[Any] = None,
        input_example: Optional[Any] = None
    ) -> None:
        """
        Log a trained model.

        Args:
            model: Trained model object
            artifact_path: Path within artifacts to save model
            registered_model_name: Optional name to register in model registry
            signature: Optional model signature
            input_example: Optional input example for signature inference
        """
        if not self._mlflow:
            return

        # Detect model type and use appropriate logging
        model_type = type(model).__module__.split('.')[0]

        try:
            if model_type == 'sklearn':
                self._mlflow.sklearn.log_model(
                    model,
                    artifact_path,
                    registered_model_name=registered_model_name,
                    signature=signature,
                    input_example=input_example
                )
            elif model_type == 'xgboost':
                self._mlflow.xgboost.log_model(
                    model,
                    artifact_path,
                    registered_model_name=registered_model_name
                )
            elif model_type == 'lightgbm':
                self._mlflow.lightgbm.log_model(
                    model,
                    artifact_path,
                    registered_model_name=registered_model_name
                )
            elif model_type in ('torch', 'pytorch'):
                self._mlflow.pytorch.log_model(
                    model,
                    artifact_path,
                    registered_model_name=registered_model_name
                )
            elif model_type in ('tensorflow', 'keras'):
                self._mlflow.keras.log_model(
                    model,
                    artifact_path,
                    registered_model_name=registered_model_name
                )
            else:
                # Generic pickle-based logging
                self._mlflow.pyfunc.log_model(
                    artifact_path,
                    python_model=model,
                    registered_model_name=registered_model_name
                )

            logger.info(f"Model logged to '{artifact_path}'")

            if registered_model_name:
                logger.info(f"Model registered as '{registered_model_name}'")

        except Exception as e:
            logger.error(f"Failed to log model: {e}")
            raise

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        """
        Log a local file as an artifact.

        Args:
            local_path: Path to local file
            artifact_path: Optional destination path in artifacts
        """
        if self._mlflow:
            self._mlflow.log_artifact(local_path, artifact_path)
            logger.debug(f"Logged artifact: {local_path}")

    def log_artifacts(self, local_dir: str, artifact_path: Optional[str] = None) -> None:
        """
        Log all files in a directory as artifacts.

        Args:
            local_dir: Path to local directory
            artifact_path: Optional destination path in artifacts
        """
        if self._mlflow:
            self._mlflow.log_artifacts(local_dir, artifact_path)
            logger.debug(f"Logged artifacts from: {local_dir}")

    def log_figure(self, figure: Any, artifact_file: str) -> None:
        """
        Log a matplotlib figure.

        Args:
            figure: Matplotlib figure object
            artifact_file: Filename for the figure
        """
        if self._mlflow:
            self._mlflow.log_figure(figure, artifact_file)

    def log_dict(self, dictionary: Dict[str, Any], artifact_file: str) -> None:
        """
        Log a dictionary as JSON artifact.

        Args:
            dictionary: Dictionary to log
            artifact_file: Filename (should end with .json)
        """
        if self._mlflow:
            self._mlflow.log_dict(dictionary, artifact_file)

    def set_tag(self, key: str, value: str) -> None:
        """
        Set a tag for the run.

        Args:
            key: Tag name
            value: Tag value
        """
        if self._mlflow:
            self._mlflow.set_tag(key, value)

    def set_tags(self, tags: Dict[str, str]) -> None:
        """
        Set multiple tags.

        Args:
            tags: Dictionary of tag names and values
        """
        if self._mlflow:
            self._mlflow.set_tags(tags)

    def get_run_info(self) -> Optional[RunInfo]:
        """Get information about the current run."""
        if not self._run:
            return None

        info = self._run.info
        return RunInfo(
            run_id=info.run_id,
            experiment_id=info.experiment_id,
            status=info.status,
            start_time=info.start_time,
            end_time=info.end_time,
            artifact_uri=info.artifact_uri
        )


def log_dataframe_info(
    df: Any,
    name: str = "dataset",
    tracker: Optional[MLflowTracker] = None
) -> Dict[str, Any]:
    """
    Log DataFrame information to MLflow.

    Args:
        df: DataFrame to log info about
        name: Name prefix for logged info
        tracker: Optional MLflowTracker (uses active run if None)

    Returns:
        Dictionary of logged information

    Example:
        >>> with MLflowTracker("exp") as tracker:
        ...     log_dataframe_info(train_df, "train", tracker)
        ...     log_dataframe_info(test_df, "test", tracker)
    """
    info = {}

    # Get basic info
    if hasattr(df, 'toPandas'):
        info["rows"] = df.count()
        info["columns"] = len(df.columns)
        info["column_names"] = df.columns
    else:
        info["rows"] = len(df)
        info["columns"] = len(df.columns)
        info["column_names"] = list(df.columns)

    # Log to MLflow
    try:
        import mlflow

        if tracker:
            tracker.log_params({
                f"{name}_rows": info["rows"],
                f"{name}_columns": info["columns"],
            })
        else:
            mlflow.log_params({
                f"{name}_rows": info["rows"],
                f"{name}_columns": info["columns"],
            })

    except Exception as e:
        logger.debug(f"Could not log to MLflow: {e}")

    return info


def log_model_metrics(
    y_true: Any,
    y_pred: Any,
    prefix: str = "",
    tracker: Optional[MLflowTracker] = None
) -> Dict[str, float]:
    """
    Calculate and log classification metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        prefix: Optional prefix for metric names
        tracker: Optional MLflowTracker

    Returns:
        Dictionary of calculated metrics
    """
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score
    )
    import numpy as np

    metrics = {
        f"{prefix}accuracy": float(accuracy_score(y_true, y_pred)),
        f"{prefix}precision": float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
        f"{prefix}recall": float(recall_score(y_true, y_pred, average='weighted', zero_division=0)),
        f"{prefix}f1": float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
    }

    # Try to calculate AUC (requires probability scores)
    try:
        if len(np.unique(y_true)) == 2:
            metrics[f"{prefix}auc"] = float(roc_auc_score(y_true, y_pred))
    except Exception:
        pass

    # Log to MLflow
    try:
        import mlflow

        if tracker:
            tracker.log_metrics(metrics)
        else:
            mlflow.log_metrics(metrics)

    except Exception as e:
        logger.debug(f"Could not log to MLflow: {e}")

    return metrics


def load_model(
    model_uri: str,
    model_type: str = "sklearn"
) -> Any:
    """
    Load a model from MLflow.

    Args:
        model_uri: Model URI (runs:/run_id/path or models:/name/version)
        model_type: Model type for loading ("sklearn", "xgboost", etc.)

    Returns:
        Loaded model

    Example:
        >>> # Load from run
        >>> model = load_model("runs:/abc123/model", "sklearn")
        >>>
        >>> # Load from registry
        >>> model = load_model("models:/my_model/Production", "sklearn")
    """
    import mlflow

    loaders = {
        "sklearn": mlflow.sklearn.load_model,
        "xgboost": mlflow.xgboost.load_model,
        "lightgbm": mlflow.lightgbm.load_model,
        "pytorch": mlflow.pytorch.load_model,
        "tensorflow": mlflow.keras.load_model,
        "keras": mlflow.keras.load_model,
        "pyfunc": mlflow.pyfunc.load_model,
    }

    loader = loaders.get(model_type, mlflow.pyfunc.load_model)

    try:
        model = loader(model_uri)
        logger.info(f"Loaded model from {model_uri}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


@contextmanager
def mlflow_run(
    experiment_name: str,
    run_name: Optional[str] = None,
    tags: Optional[Dict[str, str]] = None
):
    """
    Context manager for MLflow runs.

    Alternative to MLflowTracker for simpler usage.

    Example:
        >>> with mlflow_run("experiment", "run_name") as run:
        ...     mlflow.log_params({"lr": 0.01})
        ...     # ... train ...
        ...     mlflow.log_metrics({"acc": 0.95})
    """
    import mlflow

    mlflow.set_experiment(experiment_name)
    run = mlflow.start_run(run_name=run_name)

    if tags:
        mlflow.set_tags(tags)

    try:
        yield run
    finally:
        mlflow.end_run()


def enable_autolog(
    framework: str = "sklearn",
    log_models: bool = True,
    log_input_examples: bool = True,
    log_model_signatures: bool = True
) -> None:
    """
    Enable MLflow autologging for a framework.

    Args:
        framework: Framework to enable ("sklearn", "xgboost", "lightgbm", etc.)
        log_models: Whether to log models automatically
        log_input_examples: Whether to log input examples
        log_model_signatures: Whether to log model signatures
    """
    import mlflow

    autolog_funcs = {
        "sklearn": mlflow.sklearn.autolog,
        "xgboost": mlflow.xgboost.autolog,
        "lightgbm": mlflow.lightgbm.autolog,
        "pytorch": mlflow.pytorch.autolog,
        "tensorflow": mlflow.tensorflow.autolog,
        "keras": mlflow.keras.autolog,
        "spark": mlflow.spark.autolog,
    }

    if framework in autolog_funcs:
        autolog_funcs[framework](
            log_models=log_models,
            log_input_examples=log_input_examples,
            log_model_signatures=log_model_signatures
        )
        logger.info(f"MLflow autolog enabled for {framework}")
    else:
        logger.warning(f"Unknown framework for autolog: {framework}")
