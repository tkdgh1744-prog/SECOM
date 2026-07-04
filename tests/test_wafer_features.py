from __future__ import annotations

import unittest

import pandas as pd

from src.wafer_features import (
    heuristic_wafer_pattern_label,
    validate_wafer_columns,
    wafer_defect_features,
)


class WaferFeatureTests(unittest.TestCase):
    def test_wafer_defect_features_extracts_spatial_ratios(self) -> None:
        wafer_df = pd.DataFrame(
            {
                "wafer_id": ["W1", "W1", "W1", "W2", "W2", "W2", "W2"],
                "x": [0, 1, -1, 9, 10, -10, -9],
                "y": [0, 1, -1, 9, 10, -10, -9],
                "defect_type": ["dot", "dot", "scratch", "edge", "edge", "edge", "dot"],
            }
        )

        features = wafer_defect_features(wafer_df)

        self.assertEqual(features.shape[0], 2)
        self.assertIn("zone_ratio_center", features.columns)
        self.assertIn("zone_ratio_edge", features.columns)
        self.assertIn("defect_type_ratio_dot", features.columns)
        self.assertEqual(features.loc[features["wafer_id"] == "W1", "defect_count"].iloc[0], 3)

    def test_heuristic_pattern_label_uses_engineered_features(self) -> None:
        features = pd.DataFrame(
            {
                "zone_ratio_center": [0.7, 0.1, 0.2, 0.3],
                "zone_ratio_edge": [0.1, 0.8, 0.2, 0.2],
                "quadrant_imbalance": [0.1, 0.1, 0.7, 0.1],
            }
        )

        labels = heuristic_wafer_pattern_label(features)

        self.assertEqual(labels.tolist(), ["center", "edge", "localized", "mixed"])

    def test_validate_wafer_columns_rejects_missing_columns(self) -> None:
        with self.assertRaises(ValueError):
            validate_wafer_columns(pd.DataFrame({"wafer_id": ["W1"], "x": [1]}))


if __name__ == "__main__":
    unittest.main()
