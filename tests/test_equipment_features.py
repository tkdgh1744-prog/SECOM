from __future__ import annotations

import unittest

import pandas as pd

from src.equipment_features import (
    add_time_since_previous_event,
    equipment_event_features,
    validate_equipment_columns,
)


class EquipmentFeatureTests(unittest.TestCase):
    def test_equipment_event_features_aggregates_counts_and_failure(self) -> None:
        events_df = pd.DataFrame(
            {
                "equipment_id": ["EQ1", "EQ1", "EQ1", "EQ2"],
                "timestamp": [
                    "2026-01-01 00:00:00",
                    "2026-01-01 02:00:00",
                    "2026-01-01 03:00:00",
                    "2026-01-02 00:00:00",
                ],
                "event_type": ["alarm", "warning", "alarm", "normal"],
                "failure_label": [0, 0, 1, 0],
            }
        )

        features = equipment_event_features(events_df)
        eq1 = features.loc[features["equipment_id"] == "EQ1"].iloc[0]

        self.assertEqual(features.shape[0], 2)
        self.assertEqual(eq1["event_count"], 3)
        self.assertEqual(eq1["failure_label"], 1)
        self.assertIn("event_type_count_alarm", features.columns)
        self.assertGreater(eq1["event_rate_per_hour"], 0)

    def test_add_time_since_previous_event_computes_per_equipment_elapsed_hours(self) -> None:
        events_df = pd.DataFrame(
            {
                "equipment_id": ["EQ1", "EQ1", "EQ2"],
                "timestamp": [
                    "2026-01-01 00:00:00",
                    "2026-01-01 02:30:00",
                    "2026-01-01 04:00:00",
                ],
                "event_type": ["alarm", "warning", "normal"],
            }
        )

        enriched = add_time_since_previous_event(events_df)

        self.assertEqual(enriched.loc[0, "hours_since_previous_event"], 0)
        self.assertEqual(enriched.loc[1, "hours_since_previous_event"], 2.5)
        self.assertEqual(enriched.loc[2, "hours_since_previous_event"], 0)

    def test_validate_equipment_columns_rejects_missing_columns(self) -> None:
        with self.assertRaises(ValueError):
            validate_equipment_columns(pd.DataFrame({"equipment_id": ["EQ1"], "timestamp": ["2026-01-01"]}))


if __name__ == "__main__":
    unittest.main()
