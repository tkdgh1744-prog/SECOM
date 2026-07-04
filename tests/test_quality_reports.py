from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.quality_reports import (
    class_distribution,
    constant_features,
    dataset_overview,
    high_missing_features,
    missingness_report,
    quality_report_bundle,
)
from src.secom_data import SecomDataset


def make_dataset() -> SecomDataset:
    features = pd.DataFrame(
        {
            "feature_000": [1.0, 2.0, 3.0, 4.0],
            "feature_001": [np.nan, np.nan, 1.0, np.nan],
            "feature_002": [7.0, 7.0, 7.0, 7.0],
            "feature_003": [1.0, np.inf, 2.0, 3.0],
        }
    )
    labels = pd.Series([0, 0, 1, 1], name="target")
    timestamps = pd.to_datetime(
        pd.Series(
            [
                "2026-01-01 00:00:00",
                "2026-01-02 00:00:00",
                "2026-01-03 00:00:00",
                None,
            ]
        )
    ).rename("timestamp")
    raw_labels = pd.DataFrame({0: [-1, -1, 1, 1]})
    return SecomDataset(features=features, labels=labels, timestamps=timestamps, raw_labels=raw_labels)


class QualityReportTests(unittest.TestCase):
    def test_dataset_overview_reports_core_counts(self) -> None:
        overview = dataset_overview(make_dataset())

        self.assertEqual(overview["n_samples"], 4)
        self.assertEqual(overview["n_features"], 4)
        self.assertEqual(overview["pass_count"], 2)
        self.assertEqual(overview["fail_count"], 2)
        self.assertEqual(overview["timestamp_missing"], 1)
        self.assertEqual(overview["missing_values"], 3)
        self.assertEqual(overview["infinite_values"], 1)

    def test_class_distribution_reports_ratios(self) -> None:
        distribution = class_distribution(make_dataset().labels)

        self.assertEqual(set(distribution["class"]), {"Pass", "Fail"})
        self.assertTrue((distribution["ratio"] == 0.5).all())

    def test_missingness_and_high_missing_features(self) -> None:
        features = make_dataset().features
        report = missingness_report(features)

        self.assertEqual(report.index[0], "feature_001")
        self.assertEqual(high_missing_features(features, threshold=0.5), ["feature_001"])

    def test_constant_features_counts_nan_as_value(self) -> None:
        self.assertEqual(constant_features(make_dataset().features), ["feature_002"])

    def test_quality_report_bundle_contains_standard_reports(self) -> None:
        bundle = quality_report_bundle(make_dataset(), missing_threshold=0.5)

        self.assertIn("overview", bundle)
        self.assertIn("class_distribution", bundle)
        self.assertIn("missingness", bundle)
        self.assertIn("high_missing_features", bundle)
        self.assertIn("constant_features", bundle)


if __name__ == "__main__":
    unittest.main()
