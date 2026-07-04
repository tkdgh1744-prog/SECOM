"""Model bundle save/load utilities for SECOM prediction workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.secom_modeling import predict_with_threshold


@dataclass
class ModelBundle:
    """Serializable model bundle with prediction metadata."""

    model: Any
    threshold: float
    feature_columns: list[str]
    target_mapping: dict[int, str]
    model_name: str
    metrics: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a joblib-serializable dictionary."""
        return {
            "model": self.model,
            "threshold": self.threshold,
            "feature_columns": self.feature_columns,
            "target_mapping": self.target_mapping,
            "model_name": self.model_name,
            "metrics": self.metrics or {},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelBundle":
        """Create a ModelBundle from a dictionary."""
        required = {"model", "threshold", "feature_columns", "target_mapping", "model_name"}
        missing = sorted(required.difference(data))
        if missing:
            raise ValueError(f"Model bundle is missing keys: {missing}")

        return cls(
            model=data["model"],
            threshold=float(data["threshold"]),
            feature_columns=list(data["feature_columns"]),
            target_mapping=dict(data["target_mapping"]),
            model_name=str(data["model_name"]),
            metrics=dict(data.get("metrics") or {}),
        )


def save_model_bundle(bundle: ModelBundle, path: Path) -> Path:
    """Save a model bundle to disk with joblib."""
    import joblib

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle.to_dict(), path)
    return path


def load_model_bundle(path: Path) -> ModelBundle:
    """Load a model bundle from disk."""
    import joblib

    path = Path(path)
    data = joblib.load(path)
    if isinstance(data, ModelBundle):
        return data
    if not isinstance(data, dict):
        raise ValueError("Model bundle file must contain a dict or ModelBundle.")
    return ModelBundle.from_dict(data)


def prepare_features_for_bundle(feature_table: pd.DataFrame, bundle: ModelBundle) -> pd.DataFrame:
    """Return feature columns in the exact order expected by a model bundle."""
    missing = [column for column in bundle.feature_columns if column not in feature_table.columns]
    if missing:
        raise ValueError(f"Missing required model feature columns: {missing}")
    return feature_table.loc[:, bundle.feature_columns]


def predict_from_bundle(feature_table: pd.DataFrame, bundle: ModelBundle) -> pd.DataFrame:
    """Predict Pass/Fail labels from a saved model bundle."""
    features = prepare_features_for_bundle(feature_table, bundle)
    return predict_with_threshold(bundle.model, features, threshold=bundle.threshold).to_frame(
        index=feature_table.index
    )
