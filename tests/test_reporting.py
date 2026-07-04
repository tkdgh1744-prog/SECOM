from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.reporting import build_summary_report, dataframe_to_markdown


class ReportingTests(unittest.TestCase):
    def test_dataframe_to_markdown(self) -> None:
        df = pd.DataFrame({"metric": ["n_samples"], "value": [3]})

        markdown = dataframe_to_markdown(df)

        self.assertIn("| metric | value |", markdown)
        self.assertIn("| n_samples | 3 |", markdown)

    def test_build_summary_report_uses_available_csv_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            reports_dir = tmp_path / "reports"
            quality_dir = reports_dir / "quality"
            monitoring_dir = reports_dir / "monitoring"
            output_path = reports_dir / "summary_report.md"
            quality_dir.mkdir(parents=True)
            monitoring_dir.mkdir(parents=True)

            pd.DataFrame({"metric": ["n_samples"], "value": [3]}).to_csv(
                quality_dir / "overview.csv",
                index=False,
            )
            pd.DataFrame({"class": ["Pass", "Fail"], "count": [2, 1], "ratio": [0.67, 0.33]}).to_csv(
                quality_dir / "class_distribution.csv",
                index=False,
            )
            pd.DataFrame({"metric": ["high_risk_count"], "value": [1]}).to_csv(
                monitoring_dir / "overall_risk_summary.csv",
                index=False,
            )

            report = build_summary_report(
                reports_dir=reports_dir,
                monitoring_dir=monitoring_dir,
                output_path=output_path,
            )

            self.assertTrue(output_path.exists())
            self.assertIn("# SECOM Manufacturing Analytics Summary", report)
            self.assertIn("## Data Overview", report)
            self.assertIn("## Overall Risk Summary", report)


if __name__ == "__main__":
    unittest.main()
