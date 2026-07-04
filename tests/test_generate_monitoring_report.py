from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.generate_monitoring_report import generate_monitoring_reports, parse_group_columns


class GenerateMonitoringReportCliTests(unittest.TestCase):
    def test_parse_group_columns(self) -> None:
        self.assertEqual(parse_group_columns("wafer_id,equipment_id"), ["wafer_id", "equipment_id"])
        self.assertEqual(parse_group_columns(""), [])
        self.assertEqual(parse_group_columns(None), [])

    def test_generate_monitoring_reports_writes_csv_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            predictions_path = tmp_path / "predictions.csv"
            output_dir = tmp_path / "monitoring"

            pd.DataFrame(
                {
                    "sample_id": ["S1", "S2", "S3"],
                    "wafer_id": ["W1", "W1", "W2"],
                    "prediction": ["Pass", "Fail", "Fail"],
                    "fail_probability": [0.2, 0.7, 0.9],
                }
            ).to_csv(predictions_path, index=False)

            paths = generate_monitoring_reports(
                predictions_path=predictions_path,
                output_dir=output_dir,
                group_columns=["wafer_id"],
                high_risk_threshold=0.5,
                alert_ratio_threshold=0.5,
                top_n=2,
            )

            self.assertTrue(paths["overall"].exists())
            self.assertTrue(paths["top_risk"].exists())
            self.assertTrue(paths["group"].exists())

            top_risk = pd.read_csv(paths["top_risk"])
            group = pd.read_csv(paths["group"])

            self.assertEqual(top_risk["sample_id"].tolist(), ["S3", "S2"])
            self.assertIn("alert_flag", group.columns)


if __name__ == "__main__":
    unittest.main()
