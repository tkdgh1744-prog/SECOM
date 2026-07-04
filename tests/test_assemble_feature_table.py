from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.assemble_feature_table import assemble_from_paths, parse_key_list


class AssembleFeatureTableCliTests(unittest.TestCase):
    def test_parse_key_list(self) -> None:
        self.assertEqual(parse_key_list("wafer_id, lot_id"), ["wafer_id", "lot_id"])

    def test_assemble_from_paths_writes_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            sensor_path = tmp_path / "sensor.csv"
            wafer_path = tmp_path / "wafer.csv"
            equipment_path = tmp_path / "equipment.csv"
            output_path = tmp_path / "out" / "modeling_table.csv"
            report_dir = tmp_path / "reports"

            pd.DataFrame(
                {
                    "sample_id": ["S1", "S2", "S3"],
                    "wafer_id": ["W1", "W2", "W3"],
                    "equipment_id": ["EQ1", "EQ1", "EQ2"],
                    "feature_000": [0.1, 0.2, 0.3],
                }
            ).to_csv(sensor_path, index=False)
            pd.DataFrame(
                {
                    "wafer_id": ["W1", "W2"],
                    "defect_count": [4, 1],
                }
            ).to_csv(wafer_path, index=False)
            pd.DataFrame(
                {
                    "equipment_id": ["EQ1", "EQ2"],
                    "event_count": [5, 2],
                }
            ).to_csv(equipment_path, index=False)

            modeling_path, join_report_path, missingness_path = assemble_from_paths(
                sensor_path=sensor_path,
                wafer_path=wafer_path,
                equipment_path=equipment_path,
                sensor_wafer_keys=["wafer_id"],
                sensor_equipment_keys=["equipment_id"],
                output_path=output_path,
                report_dir=report_dir,
            )

            modeling_table = pd.read_csv(modeling_path)
            join_report = pd.read_csv(join_report_path)
            missingness = pd.read_csv(missingness_path)

            self.assertTrue(modeling_path.exists())
            self.assertTrue(join_report_path.exists())
            self.assertTrue(missingness_path.exists())
            self.assertIn("wafer_defect_count", modeling_table.columns)
            self.assertIn("equipment_event_count", modeling_table.columns)
            self.assertEqual(join_report["unmatched_rows"].tolist(), [1, 0])
            self.assertIn("wafer_defect_count", missingness["feature"].tolist())


if __name__ == "__main__":
    unittest.main()
