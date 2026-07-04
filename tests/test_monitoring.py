from __future__ import annotations

import unittest

import pandas as pd

from src.monitoring import (
    group_risk_summary,
    overall_risk_summary,
    top_risk_predictions,
    validate_prediction_columns,
)


class MonitoringTests(unittest.TestCase):
    def make_predictions(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "sample_id": ["S1", "S2", "S3", "S4"],
                "wafer_id": ["W1", "W1", "W2", "W2"],
                "equipment_id": ["EQ1", "EQ1", "EQ2", "EQ2"],
                "prediction": ["Pass", "Fail", "Pass", "Fail"],
                "fail_probability": [0.1, 0.9, 0.3, 0.8],
            }
        )

    def test_overall_risk_summary(self) -> None:
        summary = overall_risk_summary(self.make_predictions(), high_risk_threshold=0.5)

        self.assertEqual(summary["n_predictions"], 4)
        self.assertEqual(summary["predicted_fail_count"], 2)
        self.assertEqual(summary["high_risk_count"], 2)

    def test_group_risk_summary_sets_alerts(self) -> None:
        summary = group_risk_summary(
            self.make_predictions(),
            group_cols=["wafer_id"],
            high_risk_threshold=0.5,
            alert_ratio_threshold=0.4,
        )

        self.assertIn("alert_flag", summary.columns)
        self.assertTrue(summary["alert_flag"].all())
        self.assertEqual(summary["n_predictions"].tolist(), [2, 2])

    def test_top_risk_predictions_sorts_descending(self) -> None:
        top = top_risk_predictions(self.make_predictions(), top_n=2)

        self.assertEqual(top["sample_id"].tolist(), ["S2", "S4"])

    def test_validate_prediction_columns_rejects_missing_columns(self) -> None:
        with self.assertRaises(ValueError):
            validate_prediction_columns(pd.DataFrame({"prediction": ["Pass"]}))


if __name__ == "__main__":
    unittest.main()
