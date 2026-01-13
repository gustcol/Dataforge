"""
DataForge Feature Engineering

Feature engineering utilities that work across different DataFrame engines.

Features:
    - Encoders (One-Hot, Label, Target)
    - Scalers (Standard, MinMax, Robust)
    - Binning and bucketing
    - Feature interaction creation
    - Missing value handling

Best Practices:
    1. Fit transformers on training data only
    2. Apply same transformation to test data
    3. Handle missing values before encoding
    4. Consider feature importance for selection
    5. Document all transformations

Example:
    >>> from dataforge.ml import FeatureEngineer
    >>>
    >>> fe = FeatureEngineer()
    >>>
    >>> # One-hot encoding
    >>> df = fe.one_hot_encode(df, "category")
    >>>
    >>> # Standard scaling
    >>> df = fe.standard_scale(df, ["amount", "price"])
    >>>
    >>> # Binning
    >>> df = fe.bin_numeric(df, "age", bins=[0, 18, 35, 50, 65, 100])
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import logging

logger = logging.getLogger(__name__)


@dataclass
class TransformerState:
    """State for fitted transformers.

    Stores the parameters learned during fit() for later transform().
    """
    is_fitted: bool = False
    params: Dict[str, Any] = field(default_factory=dict)


class OneHotEncoder:
    """
    One-hot encoder for categorical variables.

    Converts categorical columns into binary indicator columns.

    Example:
        >>> encoder = OneHotEncoder()
        >>> encoder.fit(df, "color")
        >>> df_encoded = encoder.transform(df)
        >>> # Creates columns: color_red, color_blue, color_green, etc.
    """

    def __init__(self, drop_first: bool = False, handle_unknown: str = "ignore") -> None:
        """
        Initialize encoder.

        Args:
            drop_first: Drop first category to avoid multicollinearity
            handle_unknown: How to handle unknown categories ("ignore" or "error")
        """
        self.drop_first = drop_first
        self.handle_unknown = handle_unknown
        self._state = TransformerState()
        self._column: Optional[str] = None

    def fit(self, df: Any, column: str) -> "OneHotEncoder":
        """
        Fit encoder on training data.

        Args:
            df: Training DataFrame
            column: Column to encode

        Returns:
            Self for method chaining
        """
        self._column = column

        # Get unique categories
        if hasattr(df, 'toPandas'):
            categories = df.select(column).distinct().toPandas()[column].tolist()
        elif hasattr(df, 'to_pandas'):
            categories = df[column].unique().to_pandas().tolist()
        else:
            categories = df[column].dropna().unique().tolist()

        if self.drop_first and len(categories) > 0:
            categories = categories[1:]

        self._state.params["categories"] = categories
        self._state.is_fitted = True

        logger.info(f"OneHotEncoder fitted on '{column}' with {len(categories)} categories")
        return self

    def transform(self, df: Any) -> Any:
        """
        Transform data using fitted encoder.

        Args:
            df: DataFrame to transform

        Returns:
            Transformed DataFrame with one-hot columns
        """
        if not self._state.is_fitted:
            raise ValueError("Encoder not fitted. Call fit() first.")

        categories = self._state.params["categories"]
        column = self._column

        if hasattr(df, 'toPandas'):
            # Spark DataFrame
            from pyspark.sql import functions as F
            for cat in categories:
                df = df.withColumn(
                    f"{column}_{cat}",
                    F.when(F.col(column) == cat, 1).otherwise(0)
                )
        else:
            # Pandas or cuDF
            import pandas as pd
            for cat in categories:
                df[f"{column}_{cat}"] = (df[column] == cat).astype(int)

        return df

    def fit_transform(self, df: Any, column: str) -> Any:
        """Fit and transform in one step."""
        return self.fit(df, column).transform(df)


class LabelEncoder:
    """
    Label encoder for ordinal encoding.

    Converts categorical values to integer labels.

    Example:
        >>> encoder = LabelEncoder()
        >>> encoder.fit(df, "priority")
        >>> df_encoded = encoder.transform(df)
        >>> # "low" -> 0, "medium" -> 1, "high" -> 2
    """

    def __init__(self) -> None:
        self._state = TransformerState()
        self._column: Optional[str] = None

    def fit(self, df: Any, column: str) -> "LabelEncoder":
        """Fit encoder on training data."""
        self._column = column

        if hasattr(df, 'toPandas'):
            categories = sorted(df.select(column).distinct().toPandas()[column].dropna().tolist())
        elif hasattr(df, 'to_pandas'):
            categories = sorted(df[column].dropna().unique().to_pandas().tolist())
        else:
            categories = sorted(df[column].dropna().unique().tolist())

        self._state.params["mapping"] = {cat: i for i, cat in enumerate(categories)}
        self._state.params["inverse_mapping"] = {i: cat for i, cat in enumerate(categories)}
        self._state.is_fitted = True

        logger.info(f"LabelEncoder fitted on '{column}' with {len(categories)} categories")
        return self

    def transform(self, df: Any) -> Any:
        """Transform data using fitted encoder."""
        if not self._state.is_fitted:
            raise ValueError("Encoder not fitted. Call fit() first.")

        mapping = self._state.params["mapping"]
        column = self._column
        encoded_col = f"{column}_encoded"

        if hasattr(df, 'toPandas'):
            from pyspark.sql import functions as F
            from pyspark.sql.types import IntegerType

            mapping_expr = F.create_map([F.lit(x) for item in mapping.items() for x in item])
            df = df.withColumn(encoded_col, mapping_expr[F.col(column)].cast(IntegerType()))
        else:
            df[encoded_col] = df[column].map(mapping)

        return df

    def inverse_transform(self, df: Any, column: str) -> Any:
        """Convert encoded values back to original categories."""
        if not self._state.is_fitted:
            raise ValueError("Encoder not fitted. Call fit() first.")

        inverse_mapping = self._state.params["inverse_mapping"]

        if hasattr(df, 'toPandas'):
            from pyspark.sql import functions as F
            mapping_expr = F.create_map([F.lit(x) for item in inverse_mapping.items() for x in item])
            df = df.withColumn(f"{column}_decoded", mapping_expr[F.col(column)])
        else:
            df[f"{column}_decoded"] = df[column].map(inverse_mapping)

        return df

    def fit_transform(self, df: Any, column: str) -> Any:
        """Fit and transform in one step."""
        return self.fit(df, column).transform(df)


class StandardScaler:
    """
    Standard scaler for numerical features.

    Standardizes features by removing mean and scaling to unit variance.
    z = (x - mean) / std

    Example:
        >>> scaler = StandardScaler()
        >>> scaler.fit(df, ["amount", "quantity"])
        >>> df_scaled = scaler.transform(df)
    """

    def __init__(self, with_mean: bool = True, with_std: bool = True) -> None:
        """
        Initialize scaler.

        Args:
            with_mean: Center data by removing mean
            with_std: Scale data to unit variance
        """
        self.with_mean = with_mean
        self.with_std = with_std
        self._state = TransformerState()
        self._columns: List[str] = []

    def fit(self, df: Any, columns: List[str]) -> "StandardScaler":
        """Fit scaler on training data."""
        self._columns = columns
        stats = {}

        for col in columns:
            if hasattr(df, 'toPandas'):
                from pyspark.sql import functions as F
                row = df.agg(
                    F.mean(col).alias("mean"),
                    F.stddev(col).alias("std")
                ).collect()[0]
                stats[col] = {"mean": row["mean"], "std": row["std"]}
            else:
                stats[col] = {
                    "mean": float(df[col].mean()),
                    "std": float(df[col].std())
                }

        self._state.params["stats"] = stats
        self._state.is_fitted = True

        logger.info(f"StandardScaler fitted on {len(columns)} columns")
        return self

    def transform(self, df: Any) -> Any:
        """Transform data using fitted scaler."""
        if not self._state.is_fitted:
            raise ValueError("Scaler not fitted. Call fit() first.")

        stats = self._state.params["stats"]

        for col in self._columns:
            mean = stats[col]["mean"]
            std = stats[col]["std"]
            scaled_col = f"{col}_scaled"

            if hasattr(df, 'toPandas'):
                from pyspark.sql import functions as F
                if self.with_mean and self.with_std:
                    df = df.withColumn(scaled_col, (F.col(col) - mean) / std)
                elif self.with_mean:
                    df = df.withColumn(scaled_col, F.col(col) - mean)
                elif self.with_std:
                    df = df.withColumn(scaled_col, F.col(col) / std)
            else:
                if self.with_mean and self.with_std:
                    df[scaled_col] = (df[col] - mean) / std
                elif self.with_mean:
                    df[scaled_col] = df[col] - mean
                elif self.with_std:
                    df[scaled_col] = df[col] / std

        return df

    def fit_transform(self, df: Any, columns: List[str]) -> Any:
        """Fit and transform in one step."""
        return self.fit(df, columns).transform(df)

    def inverse_transform(self, df: Any) -> Any:
        """Convert scaled values back to original scale."""
        if not self._state.is_fitted:
            raise ValueError("Scaler not fitted. Call fit() first.")

        stats = self._state.params["stats"]

        for col in self._columns:
            mean = stats[col]["mean"]
            std = stats[col]["std"]
            scaled_col = f"{col}_scaled"
            original_col = f"{col}_unscaled"

            if hasattr(df, 'toPandas'):
                from pyspark.sql import functions as F
                df = df.withColumn(original_col, F.col(scaled_col) * std + mean)
            else:
                df[original_col] = df[scaled_col] * std + mean

        return df


class MinMaxScaler:
    """
    Min-Max scaler for numerical features.

    Scales features to a given range (default [0, 1]).
    x_scaled = (x - min) / (max - min) * (max_range - min_range) + min_range

    Example:
        >>> scaler = MinMaxScaler(feature_range=(0, 1))
        >>> scaler.fit(df, ["amount"])
        >>> df_scaled = scaler.transform(df)
    """

    def __init__(self, feature_range: tuple = (0, 1)) -> None:
        """
        Initialize scaler.

        Args:
            feature_range: Desired range of scaled features
        """
        self.feature_range = feature_range
        self._state = TransformerState()
        self._columns: List[str] = []

    def fit(self, df: Any, columns: List[str]) -> "MinMaxScaler":
        """Fit scaler on training data."""
        self._columns = columns
        stats = {}

        for col in columns:
            if hasattr(df, 'toPandas'):
                from pyspark.sql import functions as F
                row = df.agg(
                    F.min(col).alias("min"),
                    F.max(col).alias("max")
                ).collect()[0]
                stats[col] = {"min": row["min"], "max": row["max"]}
            else:
                stats[col] = {
                    "min": float(df[col].min()),
                    "max": float(df[col].max())
                }

        self._state.params["stats"] = stats
        self._state.is_fitted = True

        logger.info(f"MinMaxScaler fitted on {len(columns)} columns")
        return self

    def transform(self, df: Any) -> Any:
        """Transform data using fitted scaler."""
        if not self._state.is_fitted:
            raise ValueError("Scaler not fitted. Call fit() first.")

        stats = self._state.params["stats"]
        min_range, max_range = self.feature_range

        for col in self._columns:
            data_min = stats[col]["min"]
            data_max = stats[col]["max"]
            scaled_col = f"{col}_scaled"

            scale = (max_range - min_range) / (data_max - data_min) if data_max != data_min else 0

            if hasattr(df, 'toPandas'):
                from pyspark.sql import functions as F
                df = df.withColumn(
                    scaled_col,
                    (F.col(col) - data_min) * scale + min_range
                )
            else:
                df[scaled_col] = (df[col] - data_min) * scale + min_range

        return df

    def fit_transform(self, df: Any, columns: List[str]) -> Any:
        """Fit and transform in one step."""
        return self.fit(df, columns).transform(df)


class Binner:
    """
    Numeric binner for creating categorical features from continuous data.

    Example:
        >>> binner = Binner()
        >>> binner.fit(df, "age", bins=[0, 18, 35, 50, 65, 100])
        >>> df_binned = binner.transform(df)
        >>> # Creates age_binned column with categories
    """

    def __init__(self, labels: Optional[List[str]] = None) -> None:
        """
        Initialize binner.

        Args:
            labels: Labels for bins (length should be len(bins) - 1)
        """
        self.labels = labels
        self._state = TransformerState()
        self._column: Optional[str] = None

    def fit(
        self,
        df: Any,
        column: str,
        bins: Optional[List[float]] = None,
        n_bins: Optional[int] = None
    ) -> "Binner":
        """
        Fit binner on data.

        Args:
            df: DataFrame
            column: Column to bin
            bins: Explicit bin edges
            n_bins: Number of equal-width bins (alternative to bins)
        """
        self._column = column

        if bins is not None:
            self._state.params["bins"] = bins
        elif n_bins is not None:
            if hasattr(df, 'toPandas'):
                from pyspark.sql import functions as F
                row = df.agg(F.min(column), F.max(column)).collect()[0]
                min_val, max_val = row[0], row[1]
            else:
                min_val, max_val = float(df[column].min()), float(df[column].max())

            step = (max_val - min_val) / n_bins
            bins = [min_val + i * step for i in range(n_bins + 1)]
            self._state.params["bins"] = bins
        else:
            raise ValueError("Either bins or n_bins must be specified")

        # Generate labels if not provided
        if self.labels is None:
            bins = self._state.params["bins"]
            self.labels = [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(len(bins) - 1)]

        self._state.params["labels"] = self.labels
        self._state.is_fitted = True

        logger.info(f"Binner fitted on '{column}' with {len(self.labels)} bins")
        return self

    def transform(self, df: Any) -> Any:
        """Transform data using fitted binner."""
        if not self._state.is_fitted:
            raise ValueError("Binner not fitted. Call fit() first.")

        bins = self._state.params["bins"]
        labels = self._state.params["labels"]
        column = self._column
        binned_col = f"{column}_binned"

        if hasattr(df, 'toPandas'):
            from pyspark.sql import functions as F

            # Build CASE WHEN expression
            case_expr = None
            for i in range(len(bins) - 1):
                condition = (F.col(column) >= bins[i]) & (F.col(column) < bins[i + 1])
                if i == len(bins) - 2:  # Last bin includes upper bound
                    condition = (F.col(column) >= bins[i]) & (F.col(column) <= bins[i + 1])

                if case_expr is None:
                    case_expr = F.when(condition, labels[i])
                else:
                    case_expr = case_expr.when(condition, labels[i])

            df = df.withColumn(binned_col, case_expr)
        else:
            import pandas as pd
            df[binned_col] = pd.cut(
                df[column],
                bins=bins,
                labels=labels,
                include_lowest=True
            )

        return df

    def fit_transform(
        self,
        df: Any,
        column: str,
        bins: Optional[List[float]] = None,
        n_bins: Optional[int] = None
    ) -> Any:
        """Fit and transform in one step."""
        return self.fit(df, column, bins, n_bins).transform(df)


class FeatureEngineer:
    """
    High-level feature engineering interface.

    Provides convenient methods for common feature engineering tasks.

    Example:
        >>> fe = FeatureEngineer()
        >>>
        >>> # Encoding
        >>> df = fe.one_hot_encode(df, "category")
        >>> df = fe.label_encode(df, "priority")
        >>>
        >>> # Scaling
        >>> df = fe.standard_scale(df, ["amount", "price"])
        >>> df = fe.minmax_scale(df, ["quantity"])
        >>>
        >>> # Binning
        >>> df = fe.bin_numeric(df, "age", n_bins=5)
        >>>
        >>> # Feature creation
        >>> df = fe.create_interaction(df, ["feature1", "feature2"])
    """

    def __init__(self) -> None:
        """Initialize feature engineer."""
        self._transformers: Dict[str, Any] = {}

    def one_hot_encode(
        self,
        df: Any,
        column: str,
        drop_first: bool = False
    ) -> Any:
        """
        One-hot encode a categorical column.

        Args:
            df: DataFrame
            column: Column to encode
            drop_first: Drop first category

        Returns:
            DataFrame with one-hot encoded columns
        """
        encoder = OneHotEncoder(drop_first=drop_first)
        result = encoder.fit_transform(df, column)
        self._transformers[f"onehot_{column}"] = encoder
        return result

    def label_encode(self, df: Any, column: str) -> Any:
        """
        Label encode a categorical column.

        Args:
            df: DataFrame
            column: Column to encode

        Returns:
            DataFrame with encoded column
        """
        encoder = LabelEncoder()
        result = encoder.fit_transform(df, column)
        self._transformers[f"label_{column}"] = encoder
        return result

    def standard_scale(self, df: Any, columns: List[str]) -> Any:
        """
        Standard scale numeric columns.

        Args:
            df: DataFrame
            columns: Columns to scale

        Returns:
            DataFrame with scaled columns
        """
        scaler = StandardScaler()
        result = scaler.fit_transform(df, columns)
        self._transformers["standard_scaler"] = scaler
        return result

    def minmax_scale(
        self,
        df: Any,
        columns: List[str],
        feature_range: tuple = (0, 1)
    ) -> Any:
        """
        MinMax scale numeric columns.

        Args:
            df: DataFrame
            columns: Columns to scale
            feature_range: Output range

        Returns:
            DataFrame with scaled columns
        """
        scaler = MinMaxScaler(feature_range=feature_range)
        result = scaler.fit_transform(df, columns)
        self._transformers["minmax_scaler"] = scaler
        return result

    def bin_numeric(
        self,
        df: Any,
        column: str,
        bins: Optional[List[float]] = None,
        n_bins: Optional[int] = None,
        labels: Optional[List[str]] = None
    ) -> Any:
        """
        Bin a numeric column.

        Args:
            df: DataFrame
            column: Column to bin
            bins: Explicit bin edges
            n_bins: Number of equal-width bins
            labels: Bin labels

        Returns:
            DataFrame with binned column
        """
        binner = Binner(labels=labels)
        result = binner.fit_transform(df, column, bins, n_bins)
        self._transformers[f"binner_{column}"] = binner
        return result

    def create_interaction(
        self,
        df: Any,
        columns: List[str],
        operation: str = "multiply"
    ) -> Any:
        """
        Create interaction features between columns.

        Args:
            df: DataFrame
            columns: Columns to combine
            operation: "multiply", "add", "subtract", "divide"

        Returns:
            DataFrame with interaction column
        """
        if len(columns) != 2:
            raise ValueError("Exactly 2 columns required for interaction")

        col1, col2 = columns
        interaction_name = f"{col1}_{operation}_{col2}"

        if hasattr(df, 'toPandas'):
            from pyspark.sql import functions as F
            if operation == "multiply":
                df = df.withColumn(interaction_name, F.col(col1) * F.col(col2))
            elif operation == "add":
                df = df.withColumn(interaction_name, F.col(col1) + F.col(col2))
            elif operation == "subtract":
                df = df.withColumn(interaction_name, F.col(col1) - F.col(col2))
            elif operation == "divide":
                df = df.withColumn(interaction_name, F.col(col1) / F.col(col2))
        else:
            if operation == "multiply":
                df[interaction_name] = df[col1] * df[col2]
            elif operation == "add":
                df[interaction_name] = df[col1] + df[col2]
            elif operation == "subtract":
                df[interaction_name] = df[col1] - df[col2]
            elif operation == "divide":
                df[interaction_name] = df[col1] / df[col2]

        logger.info(f"Created interaction feature: {interaction_name}")
        return df

    def fill_missing(
        self,
        df: Any,
        columns: Optional[List[str]] = None,
        strategy: str = "mean",
        fill_value: Optional[Any] = None
    ) -> Any:
        """
        Fill missing values.

        Args:
            df: DataFrame
            columns: Columns to fill (None = all)
            strategy: "mean", "median", "mode", "constant"
            fill_value: Value for "constant" strategy

        Returns:
            DataFrame with filled values
        """
        if hasattr(df, 'toPandas'):
            from pyspark.sql import functions as F

            if columns is None:
                columns = df.columns

            for col in columns:
                if strategy == "mean":
                    fill_val = df.agg(F.mean(col)).collect()[0][0]
                elif strategy == "median":
                    fill_val = df.approxQuantile(col, [0.5], 0.01)[0]
                elif strategy == "mode":
                    fill_val = df.groupBy(col).count().orderBy(F.desc("count")).first()[0]
                elif strategy == "constant":
                    fill_val = fill_value
                else:
                    raise ValueError(f"Unknown strategy: {strategy}")

                df = df.fillna({col: fill_val})
        else:
            if columns is None:
                columns = df.columns.tolist()

            for col in columns:
                if strategy == "mean":
                    fill_val = df[col].mean()
                elif strategy == "median":
                    fill_val = df[col].median()
                elif strategy == "mode":
                    fill_val = df[col].mode()[0]
                elif strategy == "constant":
                    fill_val = fill_value
                else:
                    raise ValueError(f"Unknown strategy: {strategy}")

                df[col] = df[col].fillna(fill_val)

        logger.info(f"Filled missing values using {strategy} strategy")
        return df

    def get_transformer(self, name: str) -> Any:
        """Get a fitted transformer by name."""
        return self._transformers.get(name)

    def list_transformers(self) -> List[str]:
        """List all fitted transformers."""
        return list(self._transformers.keys())
