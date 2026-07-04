from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.run_quality_report import run_quality_report


class RunQualityReportTests(unittest.TestCase):
    def test_run_quality_report_writes_expected_csv_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            raw_dir = tmp_path / "raw"
            output_dir = tmp_path / "reports"
            raw_dir.mkdir()

            (raw_dir / "secom.data").write_text(
                "1.0 2.0 NaN\n"
                "3.0 NaN 5.0\n"
                "6.0 7.0 8.0\n",
                encoding="utf-8",
            )
            (raw_dir / "secom_labels.data").write_text(
                '-1 "19/07/2008 11:55:00"\n'
                '1 "20/07/2008 12:10:00"\n'
                '-1 "21/07/2008 13:15:00"\n',
                encoding="utf-8",
            )

            result_dir = run_quality_report(
                raw_data_dir=raw_dir,
                output_dir=output_dir,
                download=False,
                missing_threshold=0.34,
            )

            self.assertEqual(result_dir, output_dir)
            expected_files = {
                "overview.csv",
                "class_distribution.csv",
                "missingness.csv",
                "high_missing_features.csv",
                "constant_features.csv",
            }
            self.assertTrue(expected_files.issubset({path.name for path in output_dir.iterdir()}))

            overview = pd.read_csv(output_dir / "overview.csv")
            missingness = pd.read_csv(output_dir / "missingness.csv")

            self.assertIn("n_samples", overview["metric"].tolist())
            self.assertIn("feature_002", missingness["feature"].tolist())


if __name__ == "__main__":
    unittest.main()
