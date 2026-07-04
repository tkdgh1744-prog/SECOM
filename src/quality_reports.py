"""Quality report utilities for SECOM and assembled manufacturing datasets."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.secom_data import SecomDataset


def dataset_overview(dataset: SecomDataset) -> pd.Series:
    """Return a compact overview of a loaded SECOM dataset."""
    features = dataset.features
    labels = dataset.labels
    timestamps = dataset.timestamps

    return pd.Series(
        {
            "n_samples": features.shape[0],
            "n_features": features.shape[1],
            "n_labels": labels.shape[0],
            "pass_count": int((labels == 0).sum()),
            "fail_count": int((labels == 1).sum()),
            "fail_ratio": float((labels == 1).mean()),
            "date_min": timestamps.min(),
            "date_max": timestamps.max(),
            "timestamp_missing": int(timestamps.isna().sum()),
            "duplicated_rows": int(features.duplicated().sum()),
            "infinite_values": int(np.isinf(features.to_numpy(dtype=float)).sum()),
            "missing_values": int(features.isna().sum().sum()),
        },
        name="value",
    )


def class_distribution(labels: pd.Series) -> pd.DataFrame:
    """Return Pass/Fail class counts and ratios."""
    distribution = (
        labels.map({0: "Pass", 1: "Fail"})
        .value_counts()
        .rename_axis("class")
        .reset_index(name="count")
    )
    distribution["ratio"] = distribution["count"] / distribution["count"].sum()
    return distribution


def missingness_report(features: pd.DataFrame) -> pd.DataFrame:
    """Return feature-level missing counts and ratios."""
    report = pd.DataFrame(
        {
            "missing_count": features.isna().sum(),
            "missing_ratio": features.isna().mean(),
        }
    )
    return report.sort_values(["missing_ratio", "missing_count"], ascending=False)


def high_missing_features(features: pd.DataFrame, threshold: float = 0.5) -> list[str]:
    """Return feature names whose missing ratio is greater than or equal to threshold."""
    report = missingness_report(features)
    return report[report["missing_ratio"] >= threshold].index.tolist()


def constant_features(features: pd.DataFrame) -> list[str]:
    """Return feature names with a single unique value, counting NaN as a value."""
    n_unique = features.nunique(dropna=False)
    return n_unique[n_unique <= 1].index.tolist()


def quality_report_bundle(dataset: SecomDataset, missing_threshold: float = 0.5) -> dict[str, pd.DataFrame | pd.Series | list[str]]:
    """Return the standard set of data quality reports for a SECOM dataset."""
    return {
        "overview": dataset_overview(dataset),
        "class_distribution": class_distribution(dataset.labels),
        "missingness": missingness_report(dataset.features),
        "high_missing_features": high_missing_features(dataset.features, threshold=missing_threshold),
        "constant_features": constant_features(dataset.features),
    }
