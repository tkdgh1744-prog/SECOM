"""Modeling utilities for leakage-safe SECOM classification workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class PredictionResult:
    """Container for threshold-based classification output."""

    prediction: np.ndarray
    fail_probability: np.ndarray
    decision_threshold: float

    def to_frame(self, index: pd.Index | None = None) -> pd.DataFrame:
        """Return predictions as a tabular result."""
        return pd.DataFrame(
            {
                "prediction": self.prediction,
                "fail_probability": self.fail_probability,
                "decision_threshold": self.decision_threshold,
            },
            index=index,
        )


class HighMissingFeatureDropper:
    """Drop columns whose missing ratio is greater than or equal to a threshold."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def fit(self, X: pd.DataFrame, y: Any = None) -> "HighMissingFeatureDropper":
        """Learn which columns should be retained."""
        X_df = pd.DataFrame(X).copy()
        self.feature_names_in_ = X_df.columns.to_numpy()
        self.keep_columns_ = X_df.columns[X_df.isna().mean() < self.threshold].to_list()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return the retained columns."""
        self._check_is_fitted()
        X_df = pd.DataFrame(X, columns=self.feature_names_in_)
        return X_df.loc[:, self.keep_columns_]

    def fit_transform(self, X: pd.DataFrame, y: Any = None) -> pd.DataFrame:
        """Fit the transformer and return transformed data."""
        return self.fit(X, y).transform(X)

    def get_params(self, deep: bool = True) -> dict[str, float]:
        """Return parameters for sklearn compatibility."""
        return {"threshold": self.threshold}

    def set_params(self, **params: Any) -> "HighMissingFeatureDropper":
        """Set parameters for sklearn compatibility."""
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def _check_is_fitted(self) -> None:
        if not hasattr(self, "keep_columns_"):
            raise RuntimeError("HighMissingFeatureDropper must be fitted before transform.")


def make_linear_pipeline(
    missing_threshold: float = 0.5,
    low_variance_threshold: float = 1e-8,
    class_weight: str | dict[int, float] | None = None,
    random_state: int = 42,
):
    """Create a leakage-safe preprocessing and Logistic Regression pipeline."""
    from sklearn.feature_selection import VarianceThreshold
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return Pipeline(
        steps=[
            ("drop_high_missing", HighMissingFeatureDropper(threshold=missing_threshold)),
            ("imputer", SimpleImputer(strategy="median")),
            ("variance", VarianceThreshold(threshold=low_variance_threshold)),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=3000,
                    class_weight=class_weight,
                    random_state=random_state,
                ),
            ),
        ]
    )


def make_tree_pipeline(
    missing_threshold: float = 0.5,
    low_variance_threshold: float = 1e-8,
    class_weight: str | dict[int, float] | None = None,
    random_state: int = 42,
    n_estimators: int = 300,
):
    """Create a leakage-safe preprocessing and Random Forest pipeline."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_selection import VarianceThreshold
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline

    return Pipeline(
        steps=[
            ("drop_high_missing", HighMissingFeatureDropper(threshold=missing_threshold)),
            ("imputer", SimpleImputer(strategy="median")),
            ("variance", VarianceThreshold(threshold=low_variance_threshold)),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    class_weight=class_weight,
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def get_positive_proba(model: Any, X: pd.DataFrame) -> np.ndarray:
    """Return predicted probability for the Fail class when available."""
    if hasattr(model, "predict_proba"):
        proba = np.asarray(model.predict_proba(X))
        if proba.ndim != 2 or proba.shape[1] < 2:
            raise ValueError("predict_proba must return a 2D array with at least two columns.")
        return proba[:, 1]

    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(X), dtype=float)
        score_range = scores.max() - scores.min()
        if score_range == 0:
            return np.full(shape=scores.shape, fill_value=0.5, dtype=float)
        return (scores - scores.min()) / score_range

    raise AttributeError("Model does not expose predict_proba or decision_function.")


def predict_with_threshold(model: Any, X: pd.DataFrame, threshold: float = 0.5) -> PredictionResult:
    """Predict Pass/Fail labels and Fail probabilities using a decision threshold."""
    fail_probability = get_positive_proba(model, X)
    prediction = np.where(fail_probability >= threshold, "Fail", "Pass")
    return PredictionResult(
        prediction=prediction,
        fail_probability=fail_probability,
        decision_threshold=threshold,
    )


def evaluate_classifier(
    name: str,
    imbalance_method: str,
    model: Any,
    X_eval: pd.DataFrame,
    y_eval: pd.Series | np.ndarray,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Evaluate a fitted classifier with metrics focused on Fail detection."""
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        balanced_accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y_proba = get_positive_proba(model, X_eval)
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_eval, y_pred, labels=[0, 1]).ravel()

    return {
        "model": name,
        "imbalance_method": imbalance_method,
        "threshold": threshold,
        "accuracy": accuracy_score(y_eval, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_eval, y_pred),
        "fail_precision": precision_score(y_eval, y_pred, pos_label=1, zero_division=0),
        "fail_recall": recall_score(y_eval, y_pred, pos_label=1, zero_division=0),
        "fail_f1": f1_score(y_eval, y_pred, pos_label=1, zero_division=0),
        "roc_auc": roc_auc_score(y_eval, y_proba),
        "pr_auc": average_precision_score(y_eval, y_proba),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }
