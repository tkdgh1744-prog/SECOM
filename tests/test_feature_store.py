from __future__ import annotations

import unittest

import pandas as pd

from src.feature_store import (
    assemble_feature_table,
    feature_missingness_report,
    left_join_features,
    validate_key_columns,
)


class FeatureStoreTests(unittest.TestCase):
    def test_assemble_feature_table_joins_wafer_and_equipment_features(self) -> None:
        sensor_df = pd.DataFrame(
            {
                "sample_id": ["S1", "S2", "S3"],
                "wafer_id": ["W1", "W2", "W3"],
                "equipment_id": ["EQ1", "EQ1", "EQ2"],
                "feature_000": [0.1, 0.2, 0.3],
            }
        )
        wafer_df = pd.DataFrame(
            {
                "wafer_id": ["W1", "W2"],
                "defect_count": [5, 2],
                "zone_ratio_edge": [0.8, 0.1],
            }
        )
        equipment_df = pd.DataFrame(
            {
                "equipment_id": ["EQ1", "EQ2"],
                "event_count": [10, 3],
                "failure_label": [1, 0],
            }
        )

        feature_table, join_report = assemble_feature_table(
            sensor_df,
            wafer_features=wafer_df,
            equipment_features=equipment_df,
        )

        self.assertEqual(feature_table.shape[0], 3)
        self.assertIn("wafer_defect_count", feature_table.columns)
        self.assertIn("equipment_event_count", feature_table.columns)
        self.assertTrue(pd.isna(feature_table.loc[2, "wafer_defect_count"]))
        self.assertEqual(join_report["unmatched_rows"].tolist(), [1, 0])

    def test_left_join_features_prefixes_non_key_columns(self) -> None:
        base = pd.DataFrame({"id": [1, 2], "base_value": [10, 20]})
        feature = pd.DataFrame({"id": [1], "score": [0.7]})

        joined, report = left_join_features(
            base,
            feature,
            keys=["id"],
            table_name="score_table",
            feature_prefix="score_",
        )

        self.assertIn("score_score", joined.columns)
        self.assertEqual(report.unmatched_rows, 1)

    def test_validate_key_columns_rejects_missing_key(self) -> None:
        with self.assertRaises(ValueError):
            validate_key_columns(pd.DataFrame({"id": [1]}), ["missing"], "table")

    def test_feature_missingness_report_sorts_by_missing_ratio(self) -> None:
        feature_table = pd.DataFrame(
            {
                "full": [1, 2, 3],
                "some_missing": [1, None, None],
                "all_missing": [None, None, None],
            }
        )

        report = feature_missingness_report(feature_table)

        self.assertEqual(report.index[0], "all_missing")
        self.assertEqual(report.loc["some_missing", "missing_count"], 2)


if __name__ == "__main__":
    unittest.main()
