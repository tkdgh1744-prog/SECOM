from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.build_auxiliary_features import build_auxiliary_features


class BuildAuxiliaryFeaturesCliTests(unittest.TestCase):
    def test_build_auxiliary_features_writes_wafer_and_equipment_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            wafer_input = tmp_path / "wafer_inspection.csv"
            equipment_input = tmp_path / "equipment_events.csv"
            wafer_output = tmp_path / "features" / "wafer_features.csv"
            equipment_output = tmp_path / "features" / "equipment_features.csv"

            pd.DataFrame(
                {
                    "wafer_id": ["W1", "W1", "W2", "W2"],
                    "x": [0, 1, 9, -9],
                    "y": [0, 1, 9, -9],
                    "defect_type": ["dot", "scratch", "edge", "edge"],
                }
            ).to_csv(wafer_input, index=False)
            pd.DataFrame(
                {
                    "equipment_id": ["EQ1", "EQ1", "EQ2"],
                    "timestamp": ["2026-01-01 00:00:00", "2026-01-01 02:00:00", "2026-01-02 00:00:00"],
                    "event_type": ["alarm", "warning", "normal"],
                    "failure_label": [0, 1, 0],
                }
            ).to_csv(equipment_input, index=False)

            outputs = build_auxiliary_features(
                wafer_input=wafer_input,
                equipment_input=equipment_input,
                wafer_output=wafer_output,
                equipment_output=equipment_output,
                add_wafer_pattern_label=True,
            )

            wafer_features = pd.read_csv(outputs["wafer"])
            equipment_features = pd.read_csv(outputs["equipment"])

            self.assertTrue(wafer_output.exists())
            self.assertTrue(equipment_output.exists())
            self.assertIn("pattern_label", wafer_features.columns)
            self.assertIn("event_count", equipment_features.columns)
            self.assertEqual(equipment_features["event_count"].sum(), 3)

    def test_build_auxiliary_features_requires_at_least_one_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            with self.assertRaises(ValueError):
                build_auxiliary_features(
                    wafer_input=None,
                    equipment_input=None,
                    wafer_output=tmp_path / "wafer.csv",
                    equipment_output=tmp_path / "equipment.csv",
                )


if __name__ == "__main__":
    unittest.main()
