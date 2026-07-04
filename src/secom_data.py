"""Data loading utilities for the SECOM analysis project."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd


FEATURE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom.data"
LABEL_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom_labels.data"


@dataclass(frozen=True)
class SecomDataset:
    """Container for loaded SECOM feature, label, and timestamp data."""

    features: pd.DataFrame
    labels: pd.Series
    timestamps: pd.Series
    raw_labels: pd.DataFrame


def download_if_missing(url: str, destination: Path) -> Path:
    """Download a file only when it is missing or empty."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() and destination.stat().st_size > 0:
        return destination

    urlretrieve(url, destination)
    return destination


def default_secom_paths(raw_data_dir: Path) -> tuple[Path, Path]:
    """Return the default SECOM feature and label file paths."""
    raw_data_dir = Path(raw_data_dir)
    return raw_data_dir / "secom.data", raw_data_dir / "secom_labels.data"


def download_secom_dataset(raw_data_dir: Path) -> tuple[Path, Path]:
    """Download the UCI SECOM feature and label files if needed."""
    feature_path, label_path = default_secom_paths(raw_data_dir)
    download_if_missing(FEATURE_URL, feature_path)
    download_if_missing(LABEL_URL, label_path)
    return feature_path, label_path


def load_secom_data(feature_path: Path, label_path: Path) -> SecomDataset:
    """Load SECOM features, converted labels, timestamps, and raw labels.

    Original labels are mapped as follows:
    - -1 -> 0 (Pass)
    - 1 -> 1 (Fail)
    """
    feature_path = Path(feature_path)
    label_path = Path(label_path)

    features = pd.read_csv(
        feature_path,
        sep=r"\s+",
        header=None,
        na_values="NaN",
    )
    features.columns = [f"feature_{idx:03d}" for idx in range(features.shape[1])]

    raw_labels = pd.read_csv(
        label_path,
        sep=r"\s+",
        header=None,
        engine="python",
    )

    raw_label = raw_labels.iloc[:, 0].astype(int)
    labels = raw_label.replace({-1: 0, 1: 1}).rename("target")

    timestamps = raw_labels.iloc[:, 1:].astype(str).agg(" ".join, axis=1)
    timestamps = timestamps.str.replace('"', "", regex=False)
    timestamps = pd.to_datetime(timestamps, errors="coerce", dayfirst=True).rename("timestamp")

    if len(features) != len(labels):
        raise ValueError(
            f"Feature rows ({len(features)}) and label rows ({len(labels)}) do not match."
        )

    return SecomDataset(
        features=features,
        labels=labels,
        timestamps=timestamps,
        raw_labels=raw_labels,
    )


def read_optional_table(path: Path | None) -> pd.DataFrame | None:
    """Read an optional CSV/TSV/Excel table without failing when it is absent."""
    if path is None:
        return None

    path = Path(path)
    if not path.exists():
        return None

    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)

    raise ValueError(f"Unsupported optional data format: {path.suffix}")


def first_existing_path(candidates: list[Path]) -> Path | None:
    """Return the first existing path from a list of candidates."""
    return next((Path(path) for path in candidates if Path(path).exists()), None)
