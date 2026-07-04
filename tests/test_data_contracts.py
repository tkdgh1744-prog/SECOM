from __future__ import annotations

import unittest

import pandas as pd

from src.data_contracts import (
    require_columns,
    require_probability_column,
    validate_equipment_events_contract,
    validate_modeling_table_contract,
    validate_predictions_contract,
    validate_wafer_inspection_contract,
)


class DataContractTests(unittest.TestCase):
    def test_require_columns_reports_missing_columns(self) -> None:
        result = require_columns(pd.DataFrame({"a": [1]}), ["a", "b"], "table")

        self.assertFalse(result.ok)
        self.assertIn("missing columns", result.errors[0])

    def test_raise_if_failed_raises_value_error(self) -> None:
        result = require_columns(pd.DataFrame({"a": [1]}), ["missing"], "table")

        with self.assertRaises(ValueError):
            result.raise_if_failed()

    def test_validate_wafer_inspection_contract_accepts_valid_table(self) -> None:
        df = pd.DataFrame({"wafer_id": ["W1"], "x": [1.0], "y": [2.0]})

        self.assertTrue(validate_wafer_inspection_contract(df).ok)

    def test_validate_wafer_inspection_contract_rejects_non_numeric_coordinates(self) -> None:
        df = pd.DataFrame({"wafer_id": ["W1"], "x": ["bad"], "y": [2.0]})

        result = validate_wafer_inspection_contract(df)

        self.assertFalse(result.ok)
        self.assertIn("non-numeric column: x", result.errors)

    def test_validate_equipment_events_contract(self) -> None:
        df = pd.DataFrame(
            {
                "equipment_id": ["EQ1"],
                "timestamp": ["2026-01-01"],
                "event_type": ["alarm"],
            }
        )

        self.assertTrue(validate_equipment_events_contract(df).ok)

    def test_validate_predictions_contract_rejects_bad_probability_and_label(self) -> None:
        df = pd.DataFrame({"prediction": ["Pass", "Maybe"], "fail_probability": [0.1, 1.2]})

        result = validate_predictions_contract(df)

        self.assertFalse(result.ok)
        self.assertTrue(any("invalid values" in error for error in result.errors))
        self.assertTrue(any("probability out of range" in error for error in result.errors))

    def test_require_probability_column_accepts_valid_probability(self) -> None:
        result = require_probability_column(pd.DataFrame({"p": [0.0, 0.5, 1.0]}), "p", "table")

        self.assertTrue(result.ok)

    def test_validate_modeling_table_contract(self) -> None:
        df = pd.DataFrame({"feature_a": [1], "feature_b": [2]})

        self.assertTrue(validate_modeling_table_contract(df, ["feature_a", "feature_b"]).ok)
        self.assertFalse(validate_modeling_table_contract(df, ["feature_c"]).ok)


if __name__ == "__main__":
    unittest.main()
