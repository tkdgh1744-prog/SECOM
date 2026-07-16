"""Run time-aware anomaly detection on equipment sensor measurements."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.equipment_anomaly import (
    EquipmentAnomalyResult,
    build_equipment_anomaly_report,
    detect_equipment_anomalies,
)


def parse_columns(value: str | None) -> list[str]:
    """Parse a comma-separated list of sensor columns."""
    if value is None or not value.strip():
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect equipment sensor anomalies with a time-ordered CPU baseline."
    )
    parser.add_argument("--input-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/equipment_anomalies"))
    parser.add_argument("--demo", action="store_true", help="Use deterministic synthetic sensor data.")
    parser.add_argument("--equipment-id-col", default="equipment_id")
    parser.add_argument("--timestamp-col", default="timestamp")
    parser.add_argument("--sensor-columns", type=parse_columns, default=[])
    parser.add_argument("--label-col", default=None)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--contamination", type=float, default=0.05)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--integration-mode",
        choices=["unknown", "synthetic", "demo", "real"],
        default="unknown",
        help="Stamp outputs so synthetic and real manufacturing results cannot be confused.",
    )
    return parser.parse_args()


def make_demo_equipment_sensor_data(
    periods: int = 60,
    random_state: int = 42,
) -> pd.DataFrame:
    """Create deterministic hourly sensor data with late injected anomalies."""
    if periods < 12:
        raise ValueError("Demo data requires at least 12 periods.")
    rng = np.random.default_rng(random_state)
    timestamps = pd.date_range("2026-01-01", periods=periods, freq="h")
    rows: list[dict[str, object]] = []
    for equipment_offset, equipment_id in enumerate(("EQ1", "EQ2")):
        temperature = 60 + equipment_offset * 2 + rng.normal(0, 0.6, periods)
        vibration = 1.5 + equipment_offset * 0.1 + rng.normal(0, 0.08, periods)
        pressure = 100 + rng.normal(0, 0.8, periods)
        labels = np.zeros(periods, dtype=int)
        anomaly_indices = [periods - 4, periods - 2]
        temperature[anomaly_indices] += np.array([12.0, -10.0])
        vibration[anomaly_indices] += np.array([2.5, 3.0])
        pressure[anomaly_indices] += np.array([-15.0, 18.0])
        labels[anomaly_indices] = 1
        for index, timestamp in enumerate(timestamps):
            rows.append(
                {
                    "equipment_id": equipment_id,
                    "timestamp": timestamp,
                    "temperature": temperature[index],
                    "vibration": vibration[index],
                    "pressure": pressure[index],
                    "failure_label": labels[index],
                }
            )
    return pd.DataFrame(rows)


def write_equipment_anomaly_outputs(
    result: EquipmentAnomalyResult,
    output_dir: Path,
) -> dict[str, Path]:
    """Write model, scores, summaries, metrics, metadata, and report."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scores_path = output_dir / "equipment_anomaly_scores.csv"
    summary_path = output_dir / "equipment_anomaly_summary.csv"
    metrics_path = output_dir / "equipment_anomaly_metrics.csv"
    model_path = output_dir / "equipment_anomaly_model.pkl"
    metadata_path = output_dir / "equipment_anomaly_metadata.json"
    report_path = output_dir / "equipment_anomaly_report.md"

    result.scored_rows.to_csv(scores_path, index=False)
    result.summary.to_csv(summary_path, index=False)
    result.metrics.to_csv(metrics_path, index=False)
    with model_path.open("wb") as model_file:
        pickle.dump(result.bundle, model_file)
    metadata = {
        "integration_mode": str(result.scored_rows["integration_mode"].iloc[0]),
        "detector": "robust_zscore",
        "feature_columns": list(result.bundle.feature_columns),
        "threshold": result.bundle.threshold,
        "split_timestamp": result.bundle.split_timestamp,
        "contamination": result.bundle.contamination,
        "train_rows": int((result.scored_rows["split"] == "train").sum()),
        "evaluation_rows": int((result.scored_rows["split"] == "evaluation").sum()),
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    build_equipment_anomaly_report(result, report_path)
    return {
        "scores": scores_path,
        "summary": summary_path,
        "metrics": metrics_path,
        "model": model_path,
        "metadata": metadata_path,
        "report": report_path,
    }


def run_equipment_anomaly_analysis(
    input_path: Path | None = None,
    output_dir: Path = Path("outputs/equipment_anomalies"),
    demo: bool = False,
    equipment_id_col: str = "equipment_id",
    timestamp_col: str = "timestamp",
    sensor_columns: list[str] | None = None,
    label_col: str | None = None,
    train_fraction: float = 0.7,
    contamination: float = 0.05,
    random_state: int = 42,
    integration_mode: str = "unknown",
) -> dict[str, Path]:
    """Load sensor data, fit the baseline, and write all artifacts."""
    if demo:
        sensor_df = make_demo_equipment_sensor_data(random_state=random_state)
        integration_mode = "synthetic"
        label_col = label_col or "failure_label"
    else:
        if input_path is None:
            raise ValueError("Pass --input-path or use --demo.")
        sensor_df = pd.read_csv(input_path)

    result = detect_equipment_anomalies(
        sensor_df,
        sensor_columns=sensor_columns or None,
        equipment_id_col=equipment_id_col,
        timestamp_col=timestamp_col,
        label_col=label_col,
        train_fraction=train_fraction,
        contamination=contamination,
        random_state=random_state,
        integration_mode=integration_mode,
    )
    return write_equipment_anomaly_outputs(result, output_dir)


def main() -> int:
    args = parse_args()
    outputs = run_equipment_anomaly_analysis(
        input_path=args.input_path,
        output_dir=args.output_dir,
        demo=args.demo,
        equipment_id_col=args.equipment_id_col,
        timestamp_col=args.timestamp_col,
        sensor_columns=args.sensor_columns,
        label_col=args.label_col,
        train_fraction=args.train_fraction,
        contamination=args.contamination,
        random_state=args.random_state,
        integration_mode=args.integration_mode,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
