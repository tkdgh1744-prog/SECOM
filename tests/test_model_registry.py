from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

JOBLIB_AVAILABLE = importlib.util.find_spec("joblib") is not None

from src.model_registry import (
    ModelBundle,
    load_model_bundle,
    predict_from_bundle,
    prepare_features_for_bundle,
    save_model_bundle,
)


class FakeBundleModel:
    def predict_proba(self, X):
        feature_sum = X.sum(axis=1).to_numpy(dtype=float)
        proba = np.clip(feature_sum / 10.0, 0.0, 1.0)
        return np.column_stack([1.0 - proba, proba])


class ModelRegistryTests(unittest.TestCase):
    def make_bundle(self) -> ModelBundle:
        return ModelBundle(
            model=FakeBundleModel(),
            threshold=0.5,
            feature_columns=["feature_a", "feature_b"],
            target_mapping={0: "Pass", 1: "Fail"},
            model_name="fake_model",
            metrics={"fail_recall": 0.8},
        )

    @unittest.skipUnless(JOBLIB_AVAILABLE, "joblib is not installed")
    def test_save_and_load_model_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.joblib"
            bundle = self.make_bundle()

            saved_path = save_model_bundle(bundle, path)
            loaded = load_model_bundle(saved_path)

            self.assertEqual(saved_path, path)
            self.assertEqual(loaded.threshold, 0.5)
            self.assertEqual(loaded.feature_columns, ["feature_a", "feature_b"])
            self.assertEqual(loaded.metrics["fail_recall"], 0.8)

    def test_prepare_features_for_bundle_orders_columns(self) -> None:
        bundle = self.make_bundle()
        feature_table = pd.DataFrame(
            {
                "sample_id": ["S1", "S2"],
                "feature_b": [2.0, 4.0],
                "feature_a": [3.0, 1.0],
            }
        )

        prepared = prepare_features_for_bundle(feature_table, bundle)

        self.assertEqual(prepared.columns.tolist(), ["feature_a", "feature_b"])

    def test_prepare_features_for_bundle_rejects_missing_columns(self) -> None:
        with self.assertRaises(ValueError):
            prepare_features_for_bundle(pd.DataFrame({"feature_a": [1.0]}), self.make_bundle())

    def test_predict_from_bundle_uses_threshold(self) -> None:
        bundle = self.make_bundle()
        feature_table = pd.DataFrame(
            {
                "feature_a": [1.0, 4.0],
                "feature_b": [1.0, 4.0],
            },
            index=["low", "high"],
        )

        predictions = predict_from_bundle(feature_table, bundle)

        self.assertEqual(predictions["prediction"].tolist(), ["Pass", "Fail"])
        self.assertEqual(predictions.index.tolist(), ["low", "high"])


if __name__ == "__main__":
    unittest.main()
