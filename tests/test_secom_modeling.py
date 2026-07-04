from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.secom_modeling import (
    HighMissingFeatureDropper,
    get_positive_proba,
    predict_with_threshold,
)


class FakeProbaModel:
    def predict_proba(self, X):
        return np.array(
            [
                [0.8, 0.2],
                [0.4, 0.6],
                [0.1, 0.9],
            ]
        )[: len(X)]


class FakeDecisionModel:
    def __init__(self, scores):
        self.scores = np.asarray(scores, dtype=float)

    def decision_function(self, X):
        return self.scores[: len(X)]


class SecomModelingTests(unittest.TestCase):
    def test_high_missing_feature_dropper_drops_columns_at_threshold(self) -> None:
        X = pd.DataFrame(
            {
                "keep_full": [1.0, 2.0, 3.0, 4.0],
                "keep_below_threshold": [1.0, np.nan, 3.0, 4.0],
                "drop_at_threshold": [1.0, np.nan, 3.0, np.nan],
                "drop_above_threshold": [np.nan, np.nan, 3.0, np.nan],
            }
        )

        dropper = HighMissingFeatureDropper(threshold=0.5)
        transformed = dropper.fit_transform(X)

        self.assertEqual(transformed.columns.tolist(), ["keep_full", "keep_below_threshold"])

    def test_high_missing_feature_dropper_requires_fit(self) -> None:
        with self.assertRaises(RuntimeError):
            HighMissingFeatureDropper().transform(pd.DataFrame({"a": [1]}))

    def test_get_positive_proba_uses_predict_proba(self) -> None:
        X = pd.DataFrame({"feature": [1, 2, 3]})

        proba = get_positive_proba(FakeProbaModel(), X)

        np.testing.assert_allclose(proba, np.array([0.2, 0.6, 0.9]))

    def test_get_positive_proba_scales_decision_function(self) -> None:
        X = pd.DataFrame({"feature": [1, 2, 3]})

        proba = get_positive_proba(FakeDecisionModel([10, 20, 30]), X)

        np.testing.assert_allclose(proba, np.array([0.0, 0.5, 1.0]))

    def test_get_positive_proba_handles_constant_decision_scores(self) -> None:
        X = pd.DataFrame({"feature": [1, 2, 3]})

        proba = get_positive_proba(FakeDecisionModel([7, 7, 7]), X)

        np.testing.assert_allclose(proba, np.array([0.5, 0.5, 0.5]))

    def test_predict_with_threshold_returns_frame_ready_result(self) -> None:
        X = pd.DataFrame({"feature": [1, 2, 3]}, index=["a", "b", "c"])

        result = predict_with_threshold(FakeProbaModel(), X, threshold=0.6).to_frame(index=X.index)

        self.assertEqual(result["prediction"].tolist(), ["Pass", "Fail", "Fail"])
        self.assertEqual(result.index.tolist(), ["a", "b", "c"])
        np.testing.assert_allclose(result["decision_threshold"].to_numpy(), np.array([0.6, 0.6, 0.6]))


if __name__ == "__main__":
    unittest.main()
