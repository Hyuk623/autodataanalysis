"""Core analysis engine for automated EDA and simple baseline modeling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config


@dataclass
class ProblemType:
    """Detected problem type metadata."""

    type: str
    reason: str


def infer_column_types(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """Infer column roles and basic stats for each column.

    Args:
        df: Input dataframe.

    Returns:
        Mapping of column name to inferred metadata (role, n_unique, dtype).
    """

    column_info: Dict[str, Dict[str, Any]] = {}
    for col in df.columns:
        series = df[col]
        n_unique = series.nunique(dropna=True)
        dtype_str = str(series.dtype)
        role = "categorical"

        # Detect datetime
        if is_datetime64_any_dtype(series):
            role = "datetime"
        else:
            # Try parsing datetime strings only for non-numeric columns
            if not is_numeric_dtype(series):
                if series.dropna().empty:
                    parsed_datetime = False
                else:
                    try:
                        parsed = pd.to_datetime(series.dropna().iloc[:50], errors="coerce")
                        parsed_datetime = parsed.notna().mean() > 0.8
                    except Exception:
                        parsed_datetime = False
                if parsed_datetime:
                    role = "datetime"

        # Detect ID-like columns
        if role != "datetime":
            if is_numeric_dtype(series):
                role = "numeric"
                if series.dtype.kind in {"i", "u"} and n_unique == len(series):
                    role = "id"
            elif series.dtype == "object" or series.dtype == "string":
                if n_unique == len(series) and series.dropna().map(len).mean() < 50:
                    role = "id"
                else:
                    # Detect text vs categorical
                    avg_length = series.dropna().map(lambda x: len(str(x))).mean() if not series.dropna().empty else 0
                    if avg_length > 50 or n_unique > 50:
                        role = "text"
                    else:
                        role = "categorical"
            else:
                role = "categorical"

        column_info[col] = {
            "role": role,
            "n_unique": int(n_unique),
            "dtype": dtype_str,
        }

    return column_info


def infer_target_column(df: pd.DataFrame, explicit_target: Optional[str] = None) -> Tuple[Optional[str], str]:
    """Infer a candidate target column.

    Args:
        df: Input dataframe.
        explicit_target: Optional explicit target column name.

    Returns:
        Tuple of (target_column or None, reason string).
    """

    if explicit_target and explicit_target in df.columns:
        return explicit_target, "explicit target provided"

    if df.empty:
        return None, "dataframe is empty"

    column_info = infer_column_types(df)
    candidate = df.columns[-1]
    role = column_info.get(candidate, {}).get("role")
    if role in {"id", "datetime"}:
        return None, "last column looks like id/datetime"
    return candidate, "using last column as candidate target"


def infer_problem_type(df: pd.DataFrame, column_info: Dict[str, Dict[str, Any]], target_column: Optional[str]) -> ProblemType:
    """Infer the problem type based on column info and target.

    Args:
        df: Input dataframe.
        column_info: Inferred column metadata.
        target_column: Optional target column.

    Returns:
        ProblemType dataclass with inferred type and reason.
    """

    datetime_cols = [c for c, info in column_info.items() if info.get("role") == "datetime"]
    if datetime_cols:
        dt_col = datetime_cols[0]
        series = pd.to_datetime(df[dt_col], errors="coerce")
        sorted_ratio = series.dropna().is_monotonic_increasing
        if sorted_ratio:
            return ProblemType(type="time_series", reason=f"datetime column detected: {dt_col}")

    if target_column is None:
        return ProblemType(type="unsupervised", reason="no target column provided")

    target_info = column_info.get(target_column, {})
    n_unique = target_info.get("n_unique", df[target_column].nunique(dropna=True))
    if target_info.get("role") == "numeric" and n_unique > 15:
        return ProblemType(type="regression", reason="numeric target with many unique values")
    return ProblemType(type="classification", reason="categorical or low-cardinality target")


def _serialize_value(value: Any) -> Any:
    """Convert numpy/pandas objects to JSON-serializable python types."""

    if isinstance(value, (np.integer, np.int64)):
        return int(value)
    if isinstance(value, (np.floating, np.float32, np.float64)):
        return float(value)
    if isinstance(value, (np.ndarray, list, tuple)):
        return [ _serialize_value(v) for v in value ]
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return value.isoformat()
    if isinstance(value, dict):
        return {k: _serialize_value(v) for k, v in value.items()}
    return value


def run_eda_and_baseline(
    df: pd.DataFrame,
    target_column: Optional[str],
    problem_type: str,
    column_info: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Run dataset EDA and baseline modeling depending on the problem type."""

    analysis: Dict[str, Any] = {
        "dataset_overview": {},
        "column_summaries": {},
        "quality_issues": [],
        "problem_type": {"type": problem_type},
        "target_analysis": None,
        "baseline_results": None,
        "time_series_analysis": None,
        "warnings": [],
    }

    n_rows, n_cols = df.shape
    memory_bytes = df.memory_usage(deep=True).sum()
    analysis["dataset_overview"] = {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "memory_mb": round(memory_bytes / (1024 * 1024), 4),
        "head": df.head().to_dict(orient="records"),
    }

    # Missing values
    missing = df.isnull().sum()
    analysis["column_summaries"]["missing_values"] = missing.to_dict()

    # Numeric stats
    numeric_cols = [c for c, info in column_info.items() if info.get("role") == "numeric"]
    if numeric_cols:
        numeric_stats = df[numeric_cols].describe(percentiles=[0.25, 0.5, 0.75]).to_dict()
        analysis["column_summaries"]["numeric_stats"] = numeric_stats
        corr_matrix = df[numeric_cols].corr().fillna(0).to_dict()
        analysis["column_summaries"]["correlation"] = corr_matrix
    else:
        analysis["warnings"].append("No numeric columns for statistics")

    # Categorical frequencies
    categorical_cols = [c for c, info in column_info.items() if info.get("role") == "categorical"]
    freq_summary: Dict[str, Any] = {}
    for col in categorical_cols:
        freq_summary[col] = df[col].value_counts(dropna=False).head(10).to_dict()
    analysis["column_summaries"]["categorical_frequency"] = freq_summary

    # Quality issues: high missing values
    for col, count in missing.items():
        if n_rows > 0 and count / n_rows > 0.3:
            analysis["quality_issues"].append(f"Column {col} has high missing rate")

    if problem_type == "time_series":
        datetime_cols = [c for c, info in column_info.items() if info.get("role") == "datetime"]
        if datetime_cols:
            dt_col = datetime_cols[0]
            df_sorted = df.sort_values(dt_col)
            ts_analysis: Dict[str, Any] = {"datetime_column": dt_col}
            numeric_for_ts = numeric_cols
            if target_column and target_column in df_sorted.columns:
                ts_target = df_sorted[target_column]
            elif numeric_for_ts:
                ts_target = df_sorted[numeric_for_ts[0]]
                target_column = numeric_for_ts[0]
            else:
                ts_target = None
            if ts_target is not None:
                ts_analysis["rolling_mean"] = ts_target.rolling(window=5, min_periods=1).mean().tail(20).tolist()
                ts_analysis["rolling_std"] = ts_target.rolling(window=5, min_periods=1).std().fillna(0).tail(20).tolist()
                split_idx = int(len(ts_target) * 0.8)
                if split_idx == len(ts_target):
                    split_idx = max(len(ts_target) - 1, 1)
                train, test = ts_target.iloc[:split_idx], ts_target.iloc[split_idx:]
                if not test.empty:
                    # naive forecast: last observed value
                    forecast = pd.Series(train.iloc[-1], index=test.index)
                    rmse = float(np.sqrt(mean_squared_error(test, forecast)))
                    mape = float(mean_absolute_percentage_error(test, forecast)) if (test != 0).all() else None
                    ts_analysis["baseline"] = {
                        "method": "naive_last",
                        "rmse": rmse,
                        "mape": mape,
                    }
                else:
                    analysis["warnings"].append("Not enough data for time series baseline")
            analysis["time_series_analysis"] = ts_analysis
        else:
            analysis["warnings"].append("No datetime column for time series analysis")
        return _serialize_value(analysis)

    if problem_type in {"classification", "regression"} and target_column:
        target_info = column_info.get(target_column, {})
        analysis["target_analysis"] = {
            "target_column": target_column,
            "n_unique": target_info.get("n_unique", df[target_column].nunique(dropna=True)),
            "role": target_info.get("role"),
        }

        X = df.drop(columns=[target_column])
        y = df[target_column]

        cat_features = [c for c in X.columns if column_info.get(c, {}).get("role") in {"categorical", "text"}]
        num_features = [c for c in X.columns if column_info.get(c, {}).get("role") == "numeric"]

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), num_features),
                ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(handle_unknown="ignore"))]), cat_features),
            ],
            remainder="drop",
        )

        model = (
            RandomForestClassifier(n_estimators=50, random_state=config.DEFAULT_RANDOM_STATE)
            if problem_type == "classification"
            else RandomForestRegressor(n_estimators=50, random_state=config.DEFAULT_RANDOM_STATE)
        )

        clf = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

        test_size = min(config.DEFAULT_TEST_SIZE, 0.5)
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=test_size,
                random_state=config.DEFAULT_RANDOM_STATE,
                stratify=y if problem_type == "classification" else None,
            )
            clf.fit(X_train, y_train)
            preds = clf.predict(X_test)
            if problem_type == "classification":
                score = accuracy_score(y_test, preds)
                baseline_results = {"accuracy": float(score)}
            else:
                rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
                r2 = float(r2_score(y_test, preds))
                baseline_results = {"rmse": rmse, "r2": r2}

            # Feature importances if available
            model_step = clf.named_steps.get("model")
            if hasattr(model_step, "feature_importances_"):
                feature_names: List[str] = []
                if num_features:
                    feature_names.extend(num_features)
                if cat_features:
                    encoder = clf.named_steps["preprocessor"].named_transformers_["cat"].named_steps["encoder"]
                    encoded_names = encoder.get_feature_names_out(cat_features)
                    feature_names.extend(encoded_names.tolist())
                importances = model_step.feature_importances_
                importance_pairs = sorted(
                    [ (feature_names[i], float(imp)) for i, imp in enumerate(importances) ],
                    key=lambda x: x[1],
                    reverse=True,
                )
                baseline_results["feature_importance"] = importance_pairs[:10]

            analysis["baseline_results"] = baseline_results
        except Exception as exc:  # pragma: no cover - defensive
            analysis["warnings"].append(f"Baseline model failed: {exc}")

    elif problem_type == "unsupervised":
        numeric_df = df[numeric_cols]
        if not numeric_df.empty:
            scaled = StandardScaler().fit_transform(numeric_df.fillna(numeric_df.median()))
            kmeans = KMeans(n_clusters=3, random_state=config.DEFAULT_RANDOM_STATE, n_init=10)
            clusters = kmeans.fit_predict(scaled)
            analysis["baseline_results"] = {
                "method": "kmeans",
                "cluster_counts": pd.Series(clusters).value_counts().to_dict(),
            }
        else:
            analysis["warnings"].append("No numeric data for clustering")

    return _serialize_value(analysis)


def run_analysis(df: pd.DataFrame, explicit_target: Optional[str] = None) -> Dict[str, Any]:
    """High-level analysis pipeline combining inference and baseline modeling."""

    column_info = infer_column_types(df)
    target_column, target_reason = infer_target_column(df, explicit_target)
    problem = infer_problem_type(df, column_info, target_column)
    summary = run_eda_and_baseline(df, target_column, problem.type, column_info)

    return {
        "column_info": column_info,
        "target_info": {"target_column": target_column, "reason": target_reason},
        "problem_type": {"type": problem.type, "reason": problem.reason},
        "analysis_summary": summary,
    }

