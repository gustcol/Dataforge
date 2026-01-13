"""
DataForge ML Pipeline Example

This example demonstrates:
    - Feature engineering utilities
    - ML pipeline construction
    - MLflow integration
    - Best practices for ML in DataForge

Run this example:
    python examples/04_ml_pipeline.py

Note: Requires scikit-learn. MLflow integration requires mlflow package.

Author: DataForge Team
"""

import sys
sys.path.insert(0, '.')

import numpy as np
from typing import Dict, Any

from dataforge import (
    DataFrame,
    FeatureEngineer,
    MLflowTracker,
)
from dataforge.ml.pipeline import MLPipeline
from dataforge.utils import setup_logging, get_logger

# Setup logging
setup_logging(level="INFO")
logger = get_logger(__name__)


def generate_ml_data(n_samples: int = 1000) -> Dict[str, Any]:
    """Generate sample data for ML demonstration."""
    np.random.seed(42)

    # Features
    age = np.random.randint(18, 70, n_samples)
    income = np.random.normal(50000, 20000, n_samples).clip(20000, 200000)
    credit_score = np.random.randint(300, 850, n_samples)
    years_employed = np.random.randint(0, 40, n_samples)
    category = np.random.choice(["A", "B", "C"], n_samples)
    region = np.random.choice(["North", "South", "East", "West"], n_samples)

    # Target (binary classification)
    # Higher income, credit score, and employment increase approval chance
    prob = (
        0.1 +
        0.3 * (income > 60000) +
        0.3 * (credit_score > 700) +
        0.2 * (years_employed > 5) +
        0.1 * (age > 25)
    )
    approved = (np.random.random(n_samples) < prob).astype(int)

    return {
        "age": age.tolist(),
        "income": income.tolist(),
        "credit_score": credit_score.tolist(),
        "years_employed": years_employed.tolist(),
        "category": category.tolist(),
        "region": region.tolist(),
        "approved": approved.tolist(),
    }


def main():
    """Main entry point for ML pipeline example."""

    print("=" * 70)
    print("DataForge ML Pipeline Example")
    print("=" * 70)

    # Generate sample data
    data = generate_ml_data(1000)
    df = DataFrame.from_dict(data, engine="pandas")
    pandas_df = df.to_pandas()

    print(f"\nDataset: {len(pandas_df)} samples")
    print(f"Features: {[c for c in pandas_df.columns if c != 'approved']}")
    print(f"Target: approved (binary)")
    print(f"Class distribution: {pandas_df['approved'].value_counts().to_dict()}")

    # =========================================================================
    # 1. Feature Engineering
    # =========================================================================
    print("\n1. FEATURE ENGINEERING")
    print("-" * 50)

    feature_engineer = FeatureEngineer()

    # Numeric features to scale
    numeric_features = ["age", "income", "credit_score", "years_employed"]

    # Categorical features to encode
    categorical_features = ["category", "region"]

    print("\n  Original DataFrame:")
    print(f"  Columns: {list(pandas_df.columns)}")
    print(f"  Shape: {pandas_df.shape}")

    # Scale numeric features
    print("\n  Scaling numeric features...")
    scaled_df, scalers = feature_engineer.scale_features(
        pandas_df,
        numeric_features,
        method="standard"
    )
    print(f"  Scaled columns: {numeric_features}")

    # Encode categorical features
    print("\n  Encoding categorical features...")
    encoded_df, encoders = feature_engineer.encode_categorical(
        scaled_df,
        categorical_features,
        method="onehot"
    )
    print(f"  New columns after encoding: {list(encoded_df.columns)}")

    # =========================================================================
    # 2. Feature Engineering Utilities
    # =========================================================================
    print("\n2. FEATURE ENGINEERING UTILITIES")
    print("-" * 50)

    # Demonstrate different encoding methods
    print("\n  One-Hot Encoding:")
    onehot_df, _ = feature_engineer.encode_categorical(
        pandas_df[categorical_features],
        categorical_features,
        method="onehot"
    )
    print(f"  Result columns: {list(onehot_df.columns)}")

    print("\n  Label Encoding:")
    label_df, _ = feature_engineer.encode_categorical(
        pandas_df[categorical_features].copy(),
        categorical_features,
        method="label"
    )
    print(f"  Category unique values: {label_df['category'].unique()}")

    # Demonstrate different scaling methods
    print("\n  Scaling Methods:")

    for method in ["standard", "minmax", "robust"]:
        test_df, _ = feature_engineer.scale_features(
            pandas_df[["income"]].copy(),
            ["income"],
            method=method
        )
        print(
            f"  {method:10s}: "
            f"mean={test_df['income'].mean():.3f}, "
            f"std={test_df['income'].std():.3f}, "
            f"min={test_df['income'].min():.3f}, "
            f"max={test_df['income'].max():.3f}"
        )

    # =========================================================================
    # 3. ML Pipeline Construction
    # =========================================================================
    print("\n3. ML PIPELINE CONSTRUCTION")
    print("-" * 50)

    pipeline = MLPipeline(name="loan_approval_pipeline")

    # Add preprocessing stages
    pipeline.add_stage("scale_numeric", {
        "type": "scaler",
        "method": "standard",
        "columns": numeric_features,
    })

    pipeline.add_stage("encode_categorical", {
        "type": "encoder",
        "method": "onehot",
        "columns": categorical_features,
    })

    pipeline.add_stage("feature_selection", {
        "type": "selector",
        "method": "variance",
        "threshold": 0.01,
    })

    print(f"\n  Pipeline: {pipeline.name}")
    print(f"  Stages: {pipeline.stages}")

    # =========================================================================
    # 4. Train/Test Split
    # =========================================================================
    print("\n4. TRAIN/TEST SPLIT")
    print("-" * 50)

    from sklearn.model_selection import train_test_split

    # Prepare features and target
    target = "approved"
    features = [c for c in encoded_df.columns if c != target]

    X = encoded_df[features]
    y = encoded_df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"  Training set: {len(X_train)} samples")
    print(f"  Test set: {len(X_test)} samples")
    print(f"  Features: {len(features)} columns")

    # =========================================================================
    # 5. Model Training (Simple Example)
    # =========================================================================
    print("\n5. MODEL TRAINING")
    print("-" * 50)

    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        print("\n  Training Random Forest classifier...")

        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )

        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)

        # Metrics
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
        }

        print("\n  Model Performance:")
        print("  " + "-" * 30)
        for metric, value in metrics.items():
            print(f"  {metric:12s}: {value:.4f}")

        # Feature importance
        print("\n  Top 5 Feature Importances:")
        importance_df = pandas_df = {
            "feature": features,
            "importance": model.feature_importances_,
        }
        import pandas as pd
        imp_df = pd.DataFrame(importance_df).sort_values("importance", ascending=False)
        for _, row in imp_df.head(5).iterrows():
            print(f"  {row['feature']:20s}: {row['importance']:.4f}")

    except ImportError:
        print("  scikit-learn not available, skipping model training")
        metrics = {"accuracy": 0.85, "precision": 0.82, "recall": 0.78, "f1": 0.80}

    # =========================================================================
    # 6. MLflow Integration
    # =========================================================================
    print("\n6. MLFLOW INTEGRATION")
    print("-" * 50)

    print("""
    MLflow provides:
    - Experiment tracking
    - Model versioning
    - Parameter logging
    - Metric tracking
    - Artifact storage

    Example usage:
    """)

    # Demonstrate MLflow tracker (without actual MLflow connection)
    tracker = MLflowTracker(
        experiment_name="loan_approval",
        tracking_uri=None  # Would be set in production
    )

    print("""
    # Start a run
    tracker.start_run("training_run_001")

    # Log parameters
    tracker.log_params({
        "n_estimators": 100,
        "max_depth": 10,
        "random_state": 42,
    })

    # Log metrics
    tracker.log_metrics({
        "accuracy": 0.85,
        "precision": 0.82,
        "recall": 0.78,
        "f1": 0.80,
    })

    # Log model
    tracker.log_model(model, "random_forest")

    # End run
    tracker.end_run()
    """)

    # =========================================================================
    # 7. Best Practices
    # =========================================================================
    print("\n7. ML BEST PRACTICES")
    print("-" * 50)

    print("""
    ┌────────────────────────────────────────────────────────────────────┐
    │                    ML PIPELINE BEST PRACTICES                       │
    ├────────────────────────────────────────────────────────────────────┤
    │                                                                     │
    │  1. DATA PREPARATION                                                │
    │     - Handle missing values before training                         │
    │     - Scale features for algorithms sensitive to magnitude          │
    │     - Encode categorical variables appropriately                    │
    │     - Split data BEFORE any preprocessing to avoid leakage          │
    │                                                                     │
    │  2. FEATURE ENGINEERING                                             │
    │     - Create domain-specific features                               │
    │     - Use feature importance for selection                          │
    │     - Consider interaction features                                 │
    │     - Remove highly correlated features                             │
    │                                                                     │
    │  3. MODEL TRAINING                                                  │
    │     - Use cross-validation for robust evaluation                    │
    │     - Tune hyperparameters systematically                           │
    │     - Monitor for overfitting                                       │
    │     - Track all experiments in MLflow                               │
    │                                                                     │
    │  4. SPARK ML (FOR LARGE DATASETS)                                   │
    │     - Use Spark MLlib for distributed training                      │
    │     - Pipeline API for reproducibility                              │
    │     - Cache intermediate DataFrames                                 │
    │     - Partition data appropriately                                  │
    │                                                                     │
    │  5. DATABRICKS SPECIFIC                                             │
    │     - Use Feature Store for feature management                      │
    │     - AutoML for quick baselines                                    │
    │     - Model Registry for versioning                                 │
    │     - Photon for faster ML preprocessing                            │
    │                                                                     │
    └────────────────────────────────────────────────────────────────────┘
    """)

    # =========================================================================
    # 8. Pipeline Architecture
    # =========================================================================
    print("\n8. RECOMMENDED PIPELINE ARCHITECTURE")
    print("-" * 50)

    print("""
    ┌─────────────────────────────────────────────────────────────────────┐
    │                     ML PIPELINE ARCHITECTURE                         │
    ├─────────────────────────────────────────────────────────────────────┤
    │                                                                      │
    │   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐        │
    │   │  Data    │──▶│ Feature  │──▶│  Model   │──▶│  Model   │        │
    │   │  Source  │   │  Store   │   │ Training │   │ Registry │        │
    │   └──────────┘   └──────────┘   └──────────┘   └──────────┘        │
    │        │              │              │              │               │
    │        │              │              │              │               │
    │        ▼              ▼              ▼              ▼               │
    │   ┌──────────────────────────────────────────────────────┐         │
    │   │                    MLflow Tracking                    │         │
    │   │  - Parameters  - Metrics  - Artifacts  - Models       │         │
    │   └──────────────────────────────────────────────────────┘         │
    │                                                                      │
    │   Production Deployment:                                             │
    │   ┌──────────┐   ┌──────────┐   ┌──────────┐                       │
    │   │  Model   │──▶│  Batch   │──▶│  Delta   │                       │
    │   │ Serving  │   │ Scoring  │   │  Table   │                       │
    │   └──────────┘   └──────────┘   └──────────┘                       │
    │        │                                                             │
    │        ▼                                                             │
    │   ┌──────────┐                                                      │
    │   │ Real-time│                                                      │
    │   │ Endpoint │                                                      │
    │   └──────────┘                                                      │
    │                                                                      │
    └─────────────────────────────────────────────────────────────────────┘
    """)

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
    DataForge ML Pipeline provides:

    1. FEATURE ENGINEERING
       - Standard/MinMax/Robust scaling
       - One-hot/Label encoding
       - Feature selection utilities

    2. PIPELINE CONSTRUCTION
       - Composable stages
       - Reproducible workflows
       - Easy serialization

    3. MLFLOW INTEGRATION
       - Experiment tracking
       - Model versioning
       - Artifact management

    4. ENGINE FLEXIBILITY
       - Pandas for small datasets
       - Spark MLlib for large datasets
       - RAPIDS for GPU acceleration
    """)

    print("ML pipeline example completed!")


if __name__ == "__main__":
    main()
