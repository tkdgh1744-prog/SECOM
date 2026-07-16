"""Time-aware anomaly detection for equipment sensor measurements."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


VALID_INTEGRATION_MODES = {"unknown", "synthetic", "demo", "real"}
COMMON_LABEL_COLUMNS = {
    "actual_anomaly",
    "failure_label",
    "is_anomaly",
    "label",
    "target",
}


@dataclass
class EquipmentAnomalyBundle:
    """Serializable equipment anomaly model and inference metadata."""

    detector: object
    feature_columns: tuple[str, ...]
    threshold: float
    equipment_id_col: str
    timestamp_col: str
    split_timestamp: str
    contamination: float
    random_state: int


@dataclass
class EquipmentAnomalyResult:
    """Artifacts produced by a time-aware equipment anomaly analysis."""

    bundle: EquipmentAnomalyBundle
    scored_rows: pd.DataFrame
    summary: pd.DataFrame
    metrics: pd.DataFrame


@dataclass
class RobustZScoreDetector:
    """Small NumPy detector using training medians and robust scales."""

    medians: np.ndarray | None = None
    scales: np.ndarray | None = None

    def fit(self, values: pd.DataFrame | np.ndarray) -> "RobustZScoreDetector":
        array = np.asarray(values, dtype=float)
        medians = np.nanmedian(array, axis=0)
        filled = np.where(np.isnan(array), medians, array)
        mad = np.median(np.abs(filled - medians), axis=0)
        robust_scale = 1.4826 * mad
        standard_scale = np.std(filled, axis=0)
        scales = np.where(robust_scale > np.finfo(float).eps, robust_scale, standard_scale)
        self.medians = medians
        self.scales = np.where(scales > np.finfo(float).eps, scales, 1.0)
        return self

    def score_samples(self, values: pd.DataFrame | np.ndarray) -> np.ndarray:
        if self.medians is None or self.scales is None:
            raise RuntimeError("Detector must be fitted before scoring.")
        array = np.asarray(values, dtype=float)
        filled = np.where(np.isnan(array), self.medians, array)
        robust_z = (filled - self.medians) / self.scales
        return np.sqrt(np.mean(np.square(robust_z), axis=1))


def _validate_parameters(train_fraction: float, contamination: float, integration_mode: str) -> None:
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1.")
    if not 0 < contamination < 0.5:
        raise ValueError("contamination must be between 0 and 0.5.")
    if integration_mode not in VALID_INTEGRATION_MODES:
        raise ValueError(
            f"integration_mode must be one of {sorted(VALID_INTEGRATION_MODES)}."
        )


def prepare_equipment_sensor_rows(
    sensor_df: pd.DataFrame,
    equipment_id_col: str = "equipment_id",
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    """Validate and time-sort equipment sensor rows."""
    required = {equipment_id_col, timestamp_col}
    missing = sorted(required.difference(sensor_df.columns))
    if missing:
        raise ValueError(f"Missing equipment sensor columns: {missing}")
    if sensor_df.empty:
        raise ValueError("Equipment sensor data must contain at least one row.")

    prepared = sensor_df.copy()
    prepared[timestamp_col] = pd.to_datetime(prepared[timestamp_col], errors="coerce")
    if prepared[timestamp_col].isna().any():
        raise ValueError("Equipment sensor timestamps contain unparsable values.")
    if prepared[equipment_id_col].isna().any():
        raise ValueError("Equipment identifiers must not be missing.")

    return prepared.sort_values(
        [timestamp_col, equipment_id_col],
        kind="stable",
    ).reset_index(drop=True)


def select_sensor_columns(
    sensor_df: pd.DataFrame,
    sensor_columns: Sequence[str] | None = None,
    equipment_id_col: str = "equipment_id",
    timestamp_col: str = "timestamp",
    label_col: str | None = None,
) -> tuple[list[str], pd.DataFrame]:
    """Resolve numeric sensor columns and return a numeric feature frame."""
    excluded = {equipment_id_col, timestamp_col}
    if label_col:
        excluded.add(label_col)
    excluded.update(COMMON_LABEL_COLUMNS)

    if sensor_columns:
        columns = list(dict.fromkeys(sensor_columns))
        missing = sorted(set(columns).difference(sensor_df.columns))
        if missing:
            raise ValueError(f"Missing sensor feature columns: {missing}")
    else:
        columns = [
            column
            for column in sensor_df.select_dtypes(include=[np.number]).columns
            if column not in excluded
        ]

    if not columns:
        raise ValueError("At least one numeric sensor column is required.")

    numeric = sensor_df[columns].apply(pd.to_numeric, errors="coerce")
    unusable = [column for column in columns if numeric[column].notna().sum() == 0]
    if unusable:
        raise ValueError(f"Sensor columns contain no numeric values: {unusable}")
    return columns, numeric


def time_ordered_split_mask(
    timestamps: pd.Series,
    train_fraction: float = 0.7,
) -> tuple[pd.Series, pd.Timestamp]:
    """Split by unique timestamps so equal times never cross the boundary."""
    parsed = pd.to_datetime(timestamps, errors="coerce")
    if parsed.isna().any():
        raise ValueError("Timestamps contain unparsable values.")

    unique_times = pd.Index(parsed.drop_duplicates().sort_values())
    if len(unique_times) < 2:
        raise ValueError("At least two unique timestamps are required.")

    split_position = int(np.floor(len(unique_times) * train_fraction))
    split_position = min(max(split_position, 1), len(unique_times) - 1)
    split_timestamp = pd.Timestamp(unique_times[split_position])
    train_mask = parsed < split_timestamp
    if not train_mask.any() or train_mask.all():
        raise ValueError("Time split must produce both training and evaluation rows.")
    return train_mask, split_timestamp


def _classification_metrics(scored_rows: pd.DataFrame) -> pd.DataFrame:
    evaluation = scored_rows.loc[scored_rows["split"] == "evaluation"]
    if "actual_anomaly" not in evaluation.columns or evaluation.empty:
        return pd.DataFrame(
            columns=[
                "evaluation_rows",
                "true_positive",
                "false_positive",
                "true_negative",
                "false_negative",
                "precision",
                "recall",
                "f1",
            ]
        )

    actual = evaluation["actual_anomaly"].astype(bool).to_numpy()
    predicted = evaluation["is_anomaly"].astype(bool).to_numpy()
    true_positive = int(np.sum(actual & predicted))
    false_positive = int(np.sum(~actual & predicted))
    true_negative = int(np.sum(~actual & ~predicted))
    false_negative = int(np.sum(actual & ~predicted))
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2 * precision * recall / max(precision + recall, np.finfo(float).eps)

    return pd.DataFrame(
        [
            {
                "evaluation_rows": len(evaluation),
                "true_positive": true_positive,
                "false_positive": false_positive,
                "true_negative": true_negative,
                "false_negative": false_negative,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        ]
    )


def summarize_equipment_anomalies(
    scored_rows: pd.DataFrame,
    equipment_id_col: str = "equipment_id",
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    """Summarize anomaly counts and scores by equipment and time split."""
    return (
        scored_rows.groupby(
            ["integration_mode", equipment_id_col, "split"],
            dropna=False,
        )
        .agg(
            row_count=("is_anomaly", "size"),
            anomaly_count=("is_anomaly", "sum"),
            anomaly_rate=("is_anomaly", "mean"),
            mean_anomaly_score=("anomaly_score", "mean"),
            max_anomaly_score=("anomaly_score", "max"),
            first_timestamp=(timestamp_col, "min"),
            last_timestamp=(timestamp_col, "max"),
        )
        .reset_index()
        .sort_values([equipment_id_col, "split"])
    )


def detect_equipment_anomalies(
    sensor_df: pd.DataFrame,
    sensor_columns: Sequence[str] | None = None,
    equipment_id_col: str = "equipment_id",
    timestamp_col: str = "timestamp",
    label_col: str | None = None,
    train_fraction: float = 0.7,
    contamination: float = 0.05,
    random_state: int = 42,
    integration_mode: str = "unknown",
) -> EquipmentAnomalyResult:
    """Fit on the earliest time window and score all equipment sensor rows."""
    _validate_parameters(train_fraction, contamination, integration_mode)
    prepared = prepare_equipment_sensor_rows(
        sensor_df,
        equipment_id_col=equipment_id_col,
        timestamp_col=timestamp_col,
    )
    columns, numeric = select_sensor_columns(
        prepared,
        sensor_columns=sensor_columns,
        equipment_id_col=equipment_id_col,
        timestamp_col=timestamp_col,
        label_col=label_col,
    )
    train_mask, split_timestamp = time_ordered_split_mask(
        prepared[timestamp_col],
        train_fraction=train_fraction,
    )
    if int(train_mask.sum()) < 4:
        raise ValueError("At least four time-ordered training rows are required.")

    detector = RobustZScoreDetector().fit(numeric.loc[train_mask, columns])
    anomaly_scores = detector.score_samples(numeric[columns])
    train_scores = anomaly_scores[train_mask.to_numpy()]
    threshold = float(np.quantile(train_scores, 1.0 - contamination))

    scored = prepared.copy()
    scored["integration_mode"] = integration_mode
    scored["split"] = np.where(train_mask, "train", "evaluation")
    scored["anomaly_score"] = anomaly_scores.astype(float)
    scored["anomaly_threshold"] = threshold
    scored["is_anomaly"] = scored["anomaly_score"] >= threshold
    scored["anomaly_rank"] = (
        scored["anomaly_score"].rank(method="min", ascending=False).astype(int)
    )
    if label_col and label_col in scored.columns:
        labels = pd.to_numeric(scored[label_col], errors="coerce").fillna(0)
        scored["actual_anomaly"] = labels.ne(0)

    summary = summarize_equipment_anomalies(
        scored,
        equipment_id_col=equipment_id_col,
        timestamp_col=timestamp_col,
    )
    bundle = EquipmentAnomalyBundle(
        detector=detector,
        feature_columns=tuple(columns),
        threshold=threshold,
        equipment_id_col=equipment_id_col,
        timestamp_col=timestamp_col,
        split_timestamp=split_timestamp.isoformat(),
        contamination=contamination,
        random_state=random_state,
    )
    return EquipmentAnomalyResult(
        bundle=bundle,
        scored_rows=scored,
        summary=summary,
        metrics=_classification_metrics(scored),
    )


def _dataframe_to_markdown(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df.empty:
        return "_No rows._"
    display = df.head(max_rows)
    headers = [str(column) for column in display.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in display.to_numpy():
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def build_equipment_anomaly_report(
    result: EquipmentAnomalyResult,
    output_path: Path,
    top_k: int = 20,
) -> str:
    """Write a Markdown report for equipment anomaly results."""
    top_anomalies = result.scored_rows.sort_values(
        "anomaly_score",
        ascending=False,
    ).head(top_k)
    sections = [
        "# Equipment Time-Series Anomaly Analysis",
        (
            f"Integration mode: `{result.scored_rows['integration_mode'].iloc[0]}`. "
            f"Training uses timestamps before `{result.bundle.split_timestamp}`; "
            "later rows are evaluation only."
        ),
        "## Summary",
        _dataframe_to_markdown(result.summary),
        "## Evaluation Metrics",
        _dataframe_to_markdown(result.metrics),
        "## Highest Anomaly Scores",
        _dataframe_to_markdown(top_anomalies),
    ]
    report = "\n\n".join(sections) + "\n"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    return report
