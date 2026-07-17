from __future__ import annotations

import base64
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.integrated_dashboard import build_integrated_dashboard


TINY_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


class IntegratedDashboardTests(unittest.TestCase):
    def test_dashboard_integrates_available_tracks_without_cross_join(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            reports = root / "reports"
            wafer = root / "wafer"
            equipment = root / "equipment"
            output = root / "dashboard" / "index.html"
            (reports / "quality").mkdir(parents=True)
            (reports / "monitoring").mkdir(parents=True)
            (wafer / "images").mkdir(parents=True)
            equipment.mkdir(parents=True)

            pd.DataFrame({"metric": ["n_samples"], "value": [1567]}).to_csv(
                reports / "quality" / "overview.csv",
                index=False,
            )
            pd.DataFrame({"class": ["Pass", "Fail"], "count": [1463, 104]}).to_csv(
                reports / "quality" / "class_distribution.csv",
                index=False,
            )
            pd.DataFrame(
                {
                    "heuristic_pattern": ["Center", "Scratch"],
                    "wafer_count": [3, 1],
                    "mean_defect_ratio": [0.1, 0.2],
                }
            ).to_csv(wafer / "pattern_summary.csv", index=False)
            pd.DataFrame(
                {
                    "wafer_id": ["W1", "W2"],
                    "defect_ratio": [0.1, 0.3],
                    "defect_die_count": [4, 12],
                    "heuristic_pattern": ["Center", "Scratch"],
                    "source_label": ["Center", "Scratch"],
                }
            ).to_csv(wafer / "wafer_map_features.csv", index=False)
            (wafer / "images" / "pattern_summary.png").write_bytes(TINY_PNG)

            (equipment / "equipment_anomaly_metadata.json").write_text(
                json.dumps(
                    {
                        "integration_mode": "synthetic",
                        "evaluation_rows": 10,
                        "threshold": 2.1,
                    }
                ),
                encoding="utf-8",
            )
            pd.DataFrame({"f1": [0.8], "precision": [1.0], "recall": [0.67]}).to_csv(
                equipment / "equipment_anomaly_metrics.csv",
                index=False,
            )
            pd.DataFrame(
                {
                    "equipment_id": ["EQ1", "EQ1", "EQ1"],
                    "timestamp": ["2026-01-01", "2026-01-02", "2026-01-03"],
                    "anomaly_score": [5.0, 0.2, 4.0],
                    "is_anomaly": [True, False, True],
                    "split": ["train", "evaluation", "evaluation"],
                }
            ).to_csv(equipment / "equipment_anomaly_scores.csv", index=False)

            html = build_integrated_dashboard(
                reports_dir=reports,
                wafer_dir=wafer,
                equipment_dir=equipment,
                output_path=output,
                secom_mode="real",
                wafer_mode="synthetic",
            )

            self.assertTrue(output.exists())
            self.assertIn("SECOM AI", html)
            self.assertIn("Wafer Map AI", html)
            self.assertIn("Equipment AI", html)
            self.assertIn("No row-order joins", html)
            self.assertIn("data:image/png;base64", html)
            self.assertIn("SYNTHETIC", html)
            self.assertIn('"score": 4.0', html)
            self.assertIn(
                '<div class="metric-label">Evaluation anomalies</div><div class="metric-value">1</div>', html
            )

    def test_dashboard_handles_all_missing_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output = root / "dashboard.html"

            html = build_integrated_dashboard(
                reports_dir=root / "missing-reports",
                wafer_dir=root / "missing-wafer",
                equipment_dir=root / "missing-equipment",
                output_path=output,
            )

            self.assertTrue(output.exists())
            self.assertIn("0 / 3", html)
            self.assertEqual(html.count("not available"), 3)
            self.assertIn("No data available.", html)


if __name__ == "__main__":
    unittest.main()
