from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.secom_data import (
    first_existing_path,
    load_secom_data,
    read_optional_table,
)


class SecomDataLoaderTests(unittest.TestCase):
    def test_load_secom_data_maps_labels_and_timestamps(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            feature_path = tmp_path / "secom.data"
            label_path = tmp_path / "secom_labels.data"

            feature_path.write_text(
                "1.0 2.0 NaN\n"
                "4.0 NaN 6.0\n",
                encoding="utf-8",
            )
            label_path.write_text(
                '-1 "19/07/2008 11:55:00"\n'
                '1 "20/07/2008 12:10:00"\n',
                encoding="utf-8",
            )

            dataset = load_secom_data(feature_path, label_path)

            self.assertEqual(dataset.features.shape, (2, 3))
            self.assertEqual(dataset.features.columns.tolist(), ["feature_000", "feature_001", "feature_002"])
            self.assertEqual(dataset.labels.tolist(), [0, 1])
            self.assertEqual(dataset.timestamps.isna().sum(), 0)
            self.assertEqual(dataset.timestamps.dt.day.tolist(), [19, 20])

    def test_load_secom_data_rejects_row_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            feature_path = tmp_path / "secom.data"
            label_path = tmp_path / "secom_labels.data"

            feature_path.write_text("1.0 2.0\n3.0 4.0\n", encoding="utf-8")
            label_path.write_text('-1 "19/07/2008 11:55:00"\n', encoding="utf-8")

            with self.assertRaises(ValueError):
                load_secom_data(feature_path, label_path)

    def test_read_optional_table_supports_absent_csv_and_tsv(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            csv_path = tmp_path / "wafer_inspection.csv"
            tsv_path = tmp_path / "equipment_events.tsv"

            self.assertIsNone(read_optional_table(tmp_path / "missing.csv"))

            csv_path.write_text("wafer_id,x,y,defect_type\nW1,1,2,center\n", encoding="utf-8")
            tsv_path.write_text("equipment_id\ttimestamp\tevent_type\nEQ1\t2026-01-01\talarm\n", encoding="utf-8")

            csv_df = read_optional_table(csv_path)
            tsv_df = read_optional_table(tsv_path)

            self.assertIsInstance(csv_df, pd.DataFrame)
            self.assertIsInstance(tsv_df, pd.DataFrame)
            self.assertEqual(csv_df.loc[0, "defect_type"], "center")
            self.assertEqual(tsv_df.loc[0, "equipment_id"], "EQ1")

    def test_first_existing_path_returns_first_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            first = tmp_path / "missing.csv"
            second = tmp_path / "present.csv"
            second.write_text("a\n1\n", encoding="utf-8")

            self.assertEqual(first_existing_path([first, second]), second)


if __name__ == "__main__":
    unittest.main()
