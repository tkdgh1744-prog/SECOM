from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.run_pipeline import PipelinePaths, run_pipeline


class RunPipelineTests(unittest.TestCase):
    def make_paths(self, tmp_path: Path) -> PipelinePaths:
        return PipelinePaths(
            raw_data_dir=tmp_path / "raw",
            sensor_path=tmp_path / "sensor.csv",
            wafer_path=tmp_path / "wafer.csv",
            equipment_path=tmp_path / "equipment.csv",
            model_path=tmp_path / "model.joblib",
            reports_dir=tmp_path / "reports",
            features_path=tmp_path / "features" / "modeling.csv",
            predictions_path=tmp_path / "predictions" / "predictions.csv",
            monitoring_dir=tmp_path / "monitoring",
        )

    def test_run_pipeline_calls_steps_and_returns_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            paths = self.make_paths(tmp_path)
            calls = []

            def quality_step(**kwargs):
                calls.append(("quality", kwargs))
                return kwargs["output_dir"]

            def assemble_step(**kwargs):
                calls.append(("assemble", kwargs))
                return (
                    kwargs["output_path"],
                    kwargs["report_dir"] / "feature_join_report.csv",
                    kwargs["report_dir"] / "feature_missingness_report.csv",
                )

            def prediction_step(**kwargs):
                calls.append(("prediction", kwargs))
                return kwargs["output_path"]

            def monitoring_step(**kwargs):
                calls.append(("monitoring", kwargs))
                return {"overall": kwargs["output_dir"] / "overall_risk_summary.csv"}

            result = run_pipeline(
                paths=paths,
                id_columns=["sample_id"],
                monitoring_group_columns=["wafer_id"],
                quality_step=quality_step,
                assemble_step=assemble_step,
                prediction_step=prediction_step,
                monitoring_step=monitoring_step,
            )

            self.assertEqual([name for name, _ in calls], ["quality", "assemble", "prediction", "monitoring"])
            self.assertEqual(result.feature_table_path, paths.features_path)
            self.assertEqual(result.predictions_path, paths.predictions_path)
            self.assertIn("overall", result.monitoring_paths)
            self.assertEqual(calls[2][1]["id_columns"], ["sample_id"])
            self.assertEqual(calls[3][1]["group_columns"], ["wafer_id"])

    def test_run_pipeline_can_skip_quality_and_prediction(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            paths = self.make_paths(tmp_path)
            calls = []

            def assemble_step(**kwargs):
                calls.append(("assemble", kwargs))
                return (
                    kwargs["output_path"],
                    kwargs["report_dir"] / "feature_join_report.csv",
                    kwargs["report_dir"] / "feature_missingness_report.csv",
                )

            result = run_pipeline(
                paths=paths,
                skip_quality_report=True,
                skip_prediction=True,
                assemble_step=assemble_step,
            )

            self.assertEqual([name for name, _ in calls], ["assemble"])
            self.assertIsNone(result.predictions_path)
            self.assertEqual(result.monitoring_paths, {})


if __name__ == "__main__":
    unittest.main()
