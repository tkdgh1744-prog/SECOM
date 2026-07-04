from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.predict_with_model import parse_id_columns, run_prediction
from src.model_registry import ModelBundle, save_model_bundle


JOBLIB_AVAILABLE = importlib.util.find_spec("joblib") is not None


class FakePredictionModel:
    def predict_proba(self, X):
        proba = np.clip(X.sum(axis=1).to_numpy(dtype=float) / 10.0, 0.0, 1.0)
        return np.column_stack([1.0 - proba, proba])


class PredictWithModelCliTests(unittest.TestCase):
    def test_parse_id_columns(self) -> None:
        self.assertEqual(parse_id_columns("sample_id, wafer_id"), ["sample_id", "wafer_id"])
        self.assertEqual(parse_id_columns(""), [])
        self.assertEqual(parse_id_columns(None), [])

    @unittest.skipUnless(JOBLIB_AVAILABLE, "joblib is not installed")
    def test_run_prediction_writes_prediction_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            model_path = tmp_path / "model.joblib"
            features_path = tmp_path / "features.csv"
            output_path = tmp_path / "predictions" / "predictions.csv"

            bundle = ModelBundle(
                model=FakePredictionModel(),
                threshold=0.5,
                feature_columns=["feature_a", "feature_b"],
                target_mapping={0: "Pass", 1: "Fail"},
                model_name="fake",
                metrics={"fail_recall": 1.0},
            )
            save_model_bundle(bundle, model_path)

            pd.DataFrame(
                {
                    "sample_id": ["S1", "S2"],
                    "feature_b": [1.0, 4.0],
                    "feature_a": [1.0, 4.0],
                }
            ).to_csv(features_path, index=False)

            result_path = run_prediction(
                model_path=model_path,
                features_path=features_path,
                output_path=output_path,
                id_columns=["sample_id"],
            )

            predictions = pd.read_csv(result_path)

            self.assertEqual(predictions["sample_id"].tolist(), ["S1", "S2"])
            self.assertEqual(predictions["prediction"].tolist(), ["Pass", "Fail"])
            self.assertIn("fail_probability", predictions.columns)

    @unittest.skipUnless(JOBLIB_AVAILABLE, "joblib is not installed")
    def test_run_prediction_rejects_missing_id_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            model_path = tmp_path / "model.joblib"
            features_path = tmp_path / "features.csv"

            save_model_bundle(
                ModelBundle(
                    model=FakePredictionModel(),
                    threshold=0.5,
                    feature_columns=["feature_a"],
                    target_mapping={0: "Pass", 1: "Fail"},
                    model_name="fake",
                ),
                model_path,
            )
            pd.DataFrame({"feature_a": [1.0]}).to_csv(features_path, index=False)

            with self.assertRaises(ValueError):
                run_prediction(
                    model_path=model_path,
                    features_path=features_path,
                    output_path=tmp_path / "predictions.csv",
                    id_columns=["missing_id"],
                )


if __name__ == "__main__":
    unittest.main()
