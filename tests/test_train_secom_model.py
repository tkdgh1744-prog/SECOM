from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.train_secom_model import resolve_secom_paths, write_test_predictions


class FakeModel:
    def predict_proba(self, X):
        probabilities = np.array([0.2, 0.8])[: len(X)]
        return np.column_stack([1.0 - probabilities, probabilities])


class TrainSecomModelTests(unittest.TestCase):
    def test_resolve_secom_paths_uses_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_data_dir = Path(tmpdir)
            feature_path = raw_data_dir / "secom.data"
            label_path = raw_data_dir / "secom_labels.data"
            feature_path.write_text("1 2\n", encoding="utf-8")
            label_path.write_text("-1 now\n", encoding="utf-8")

            resolved = resolve_secom_paths(raw_data_dir=raw_data_dir)

            self.assertEqual(resolved, (feature_path, label_path))

    def test_resolve_secom_paths_rejects_missing_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(FileNotFoundError):
                resolve_secom_paths(raw_data_dir=Path(tmpdir))

    def test_write_test_predictions_includes_labels_and_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "predictions.csv"
            X_test = pd.DataFrame({"feature_000": [1.0, 2.0]}, index=[10, 11])
            y_test = pd.Series([0, 1], index=X_test.index)
            timestamps = pd.Series(pd.to_datetime(["2026-01-01", "2026-01-02"]), index=X_test.index)

            written_path = write_test_predictions(
                output_path=output_path,
                model=FakeModel(),
                X_test=X_test,
                y_test=y_test,
                timestamps=timestamps,
                threshold=0.5,
            )

            predictions = pd.read_csv(written_path)

            self.assertEqual(written_path, output_path)
            self.assertEqual(predictions["prediction"].tolist(), ["Pass", "Fail"])
            self.assertEqual(predictions["actual_name"].tolist(), ["Pass", "Fail"])
            self.assertEqual(predictions["decision_threshold"].tolist(), [0.5, 0.5])


if __name__ == "__main__":
    unittest.main()
