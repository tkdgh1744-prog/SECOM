from __future__ import annotations

import json
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.analyze_equipment_anomalies import (
    make_demo_equipment_sensor_data,
    run_equipment_anomaly_analysis,
)
from src.equipment_anomaly import (
    detect_equipment_anomalies,
    time_ordered_split_mask,
)


class EquipmentAnomalyTests(unittest.TestCase):
    def test_time_split_keeps_equal_timestamps_together(self) -> None:
        timestamps = pd.Series(
            pd.to_datetime(
                [
                    "2026-01-01 00:00",
                    "2026-01-01 00:00",
                    "2026-01-01 01:00",
                    "2026-01-01 01:00",
                    "2026-01-01 02:00",
                    "2026-01-01 02:00",
                ]
            )
        )

        train_mask, split_timestamp = time_ordered_split_mask(
            timestamps,
            train_fraction=0.67,
        )

        self.assertEqual(split_timestamp, pd.Timestamp("2026-01-01 02:00"))
        self.assertTrue((timestamps[train_mask] < split_timestamp).all())
        self.assertTrue((timestamps[~train_mask] >= split_timestamp).all())

    def test_detector_flags_large_late_anomaly(self) -> None:
        periods = 24
        frame = pd.DataFrame(
            {
                "equipment_id": ["EQ1"] * periods,
                "timestamp": pd.date_range("2026-01-01", periods=periods, freq="h"),
                "temperature": np.linspace(60.0, 61.0, periods),
                "vibration": np.linspace(1.0, 1.1, periods),
                "failure_label": [0] * (periods - 1) + [1],
            }
        )
        frame.loc[periods - 1, ["temperature", "vibration"]] = [100.0, 8.0]

        result = detect_equipment_anomalies(
            frame,
            sensor_columns=["temperature", "vibration"],
            label_col="failure_label",
            train_fraction=0.7,
            contamination=0.1,
            integration_mode="synthetic",
        )

        last = result.scored_rows.iloc[-1]
        self.assertEqual(last["split"], "evaluation")
        self.assertTrue(bool(last["is_anomaly"]))
        self.assertEqual(int(last["anomaly_rank"]), 1)
        self.assertEqual(result.bundle.feature_columns, ("temperature", "vibration"))
        self.assertEqual(result.metrics.loc[0, "true_positive"], 1)

    def test_detector_rejects_missing_numeric_sensors(self) -> None:
        frame = pd.DataFrame(
            {
                "equipment_id": ["EQ1"] * 6,
                "timestamp": pd.date_range("2026-01-01", periods=6, freq="h"),
                "state": ["normal"] * 6,
            }
        )

        with self.assertRaisesRegex(ValueError, "numeric sensor"):
            detect_equipment_anomalies(frame)

    def test_auto_sensor_selection_excludes_common_label_columns(self) -> None:
        periods = 12
        frame = pd.DataFrame(
            {
                "equipment_id": ["EQ1"] * periods,
                "timestamp": pd.date_range("2026-01-01", periods=periods, freq="h"),
                "temperature": np.linspace(60.0, 61.0, periods),
                "failure_label": [0] * (periods - 1) + [1],
            }
        )
        result = detect_equipment_anomalies(
            frame,
            train_fraction=0.7,
            contamination=0.1,
        )

        self.assertEqual(result.bundle.feature_columns, ("temperature",))

    def test_demo_analysis_writes_machine_readable_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "equipment"
            outputs = run_equipment_anomaly_analysis(
                output_dir=output_dir,
                demo=True,
                sensor_columns=["temperature", "vibration", "pressure"],
                train_fraction=0.7,
                contamination=0.05,
            )

            for path in outputs.values():
                self.assertTrue(path.exists())
            scores = pd.read_csv(outputs["scores"])
            metadata = json.loads(outputs["metadata"].read_text(encoding="utf-8"))
            self.assertEqual(set(scores["integration_mode"]), {"synthetic"})
            self.assertGreater(int(scores["is_anomaly"].sum()), 0)
            self.assertEqual(metadata["integration_mode"], "synthetic")
            self.assertGreater(metadata["train_rows"], metadata["evaluation_rows"])
            with outputs["model"].open("rb") as model_file:
                bundle = pickle.load(model_file)
            self.assertEqual(bundle.feature_columns, ("temperature", "vibration", "pressure"))
            self.assertEqual(bundle.detector.score_samples([[60.0, 1.5, 100.0]]).shape, (1,))

    def test_demo_data_injects_labeled_late_anomalies(self) -> None:
        demo = make_demo_equipment_sensor_data(periods=20)

        self.assertEqual(len(demo), 40)
        self.assertEqual(int(demo["failure_label"].sum()), 4)
        anomaly_times = demo.loc[demo["failure_label"] == 1, "timestamp"]
        self.assertGreater(anomaly_times.min(), demo["timestamp"].quantile(0.7))


if __name__ == "__main__":
    unittest.main()
