from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.wafer_dataset import (
    inspect_wm811k_dataset,
    sha256_file,
    write_wm811k_preflight,
)


class WaferDatasetPreflightTests(unittest.TestCase):
    def _write_dataset(self, path: Path) -> None:
        pd.DataFrame(
            {
                "waferMap": [
                    np.array([[0, 1, 0], [1, 2, 1]]),
                    np.array([[0, 1, 0], [2, 1, 1]]),
                    np.array([[0, 1, 0], [1, 1, 2]]),
                ],
                "failureType": [["Center"], ["Center"], ["Scratch"]],
                "lotName": [["Lot-1"], ["Lot-1"], ["Lot-2"]],
                "waferIndex": [1, 2, 3],
            }
        ).to_pickle(path)

    def test_complete_provenance_is_ready_for_grouped_training(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "LSWMD.pkl"
            self._write_dataset(path)

            report = inspect_wm811k_dataset(
                path,
                id_col="waferIndex",
                sample_records=3,
                source_uri="https://example.test/wm811k",
                license_note="Research dataset terms verified.",
                compute_sha256=True,
            )

            self.assertEqual(report["schema"]["row_count"], 3)
            self.assertEqual(report["labels"]["counts"], {"Center": 2, "Scratch": 1})
            self.assertEqual(report["groups"]["group_count"], 2)
            self.assertEqual(report["map_sample"]["valid_2d_maps"], 3)
            self.assertTrue(report["provenance"]["complete"])
            self.assertTrue(report["ready_for_grouped_training"])

    def test_missing_provenance_and_group_are_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "maps.pkl"
            pd.DataFrame(
                {
                    "waferMap": [np.ones((2, 2), dtype=int)],
                    "failureType": [["Center"]],
                }
            ).to_pickle(path)

            report = inspect_wm811k_dataset(path, sample_records=1)

            self.assertFalse(report["provenance"]["complete"])
            self.assertFalse(report["ready_for_grouped_training"])
            self.assertTrue(any("lot/group" in warning for warning in report["warnings"]))
            self.assertTrue(
                any("Fewer than two labeled" in warning for warning in report["warnings"])
            )

    def test_missing_wafer_map_column_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bad.pkl"
            pd.DataFrame({"other": [1]}).to_pickle(path)

            with self.assertRaisesRegex(ValueError, "Missing wafer map column"):
                inspect_wm811k_dataset(path)

    def test_report_writer_and_checksum_are_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "data.bin"
            input_path.write_bytes(b"wafer-data")
            self.assertEqual(
                sha256_file(input_path),
                "1bd2f8cf09b267102a3d8e19e12b1bebdd166c46a446aea3f2fbadb5038ec680",
            )
            output_path = Path(tmpdir) / "report.json"
            write_wm811k_preflight({"ready": True}, output_path)
            self.assertEqual(json.loads(output_path.read_text(encoding="utf-8")), {"ready": True})


if __name__ == "__main__":
    unittest.main()
