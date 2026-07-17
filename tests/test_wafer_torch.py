from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.analyze_wafer_maps import make_demo_records
from src.wafer_map_analysis import load_wm811k_dataframe
from src.wafer_torch import (
    export_torch_model_to_onnx,
    grouped_train_test_indices,
    load_torch_model_bundle,
    onnx_is_available,
    onnxruntime_is_available,
    prepare_wafer_images,
    save_torch_model_bundle,
    torch_is_available,
    train_torch_autoencoder_anomaly_scores,
    train_torch_cnn_pattern_classifier,
)


class WaferTorchDataTests(unittest.TestCase):
    def test_prepare_wafer_images_returns_nchw_float32(self) -> None:
        images = prepare_wafer_images(make_demo_records(), resize_to=24)

        self.assertEqual(images.shape, (6, 1, 24, 24))
        self.assertEqual(images.dtype, np.float32)
        self.assertTrue(set(np.unique(images)).issubset({0.0, 1.0}))

    def test_grouped_split_never_crosses_lots(self) -> None:
        records = make_demo_records(variants_per_pattern=3)
        labels = [str(record.label) for record in records]

        train_indices, test_indices = grouped_train_test_indices(
            records,
            labels=labels,
            test_fraction=0.34,
            random_state=7,
        )

        train_groups = {records[index].group_id for index in train_indices}
        test_groups = {records[index].group_id for index in test_indices}
        self.assertFalse(train_groups.intersection(test_groups))
        self.assertEqual({labels[index] for index in train_indices}, set(labels))

    def test_demo_training_variants_preserve_all_classes_per_lot(self) -> None:
        records = make_demo_records(variants_per_pattern=3)
        classes_by_group: dict[str, set[str]] = {}
        for record in records:
            classes_by_group.setdefault(str(record.group_id), set()).add(str(record.label))

        self.assertEqual(len(records), 18)
        self.assertEqual(len(classes_by_group), 3)
        self.assertTrue(all(len(classes) == 6 for classes in classes_by_group.values()))

    def test_wm811k_loader_reads_lot_as_group(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "wm811k.pkl"
            pd.DataFrame(
                {
                    "waferMap": [np.array([[0, 1], [1, 2]])],
                    "failureType": [["Center"]],
                    "lotName": [["Lot-17"]],
                }
            ).to_pickle(path)

            records = load_wm811k_dataframe(path)

            self.assertEqual(records[0].label, "Center")
            self.assertEqual(records[0].group_id, "Lot-17")


@unittest.skipUnless(torch_is_available(), "PyTorch is not installed")
class WaferTorchTrainingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.records = make_demo_records(variants_per_pattern=3)

    def test_cnn_trains_and_saves_portable_bundle(self) -> None:
        result = train_torch_cnn_pattern_classifier(
            self.records,
            [str(record.label) for record in self.records],
            epochs=1,
            batch_size=18,
            device="cpu",
        )

        self.assertEqual(result["device"], "cpu")
        self.assertEqual(len(result["classes"]), 6)
        self.assertGreater(result["parameter_count"], 0)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_torch_model_bundle(
                result,
                Path(tmpdir) / "wafer_cnn.pt",
                model_type="wafer_pattern_cnn",
            )
            self.assertTrue(path.exists())
            loaded = load_torch_model_bundle(path, device="cpu")
            self.assertEqual(loaded["model_type"], "wafer_pattern_cnn")
            self.assertEqual(loaded["classes"], result["classes"])
            self.assertEqual(loaded["parameter_count"], result["parameter_count"])

    def test_autoencoder_trains_on_grouped_subset_and_scores_all_maps(self) -> None:
        result = train_torch_autoencoder_anomaly_scores(
            self.records,
            epochs=1,
            batch_size=18,
            device="cpu",
        )

        scores = result["scores"]
        self.assertEqual(len(scores), len(self.records))
        self.assertEqual(set(scores["split"]), {"train", "test"})
        self.assertTrue((scores["autoencoder_reconstruction_error"] >= 0).all())

    @unittest.skipUnless(
        onnx_is_available() and onnxruntime_is_available(),
        "ONNX or ONNX Runtime is not installed",
    )
    def test_cnn_exports_valid_onnx(self) -> None:
        import onnx
        import onnxruntime as ort

        result = train_torch_cnn_pattern_classifier(
            self.records,
            [str(record.label) for record in self.records],
            epochs=1,
            batch_size=18,
            device="cpu",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = export_torch_model_to_onnx(result, Path(tmpdir) / "wafer_cnn.onnx")
            model = onnx.load(path)
            onnx.checker.check_model(model)
            self.assertTrue(path.exists())
            session = ort.InferenceSession(
                str(path),
                providers=["CPUExecutionProvider"],
            )
            inputs = prepare_wafer_images(self.records[:2], resize_to=32)
            outputs = session.run(None, {"wafer_map": inputs})
            self.assertEqual(len(outputs), 1)
            self.assertEqual(outputs[0].shape, (2, 6))


if __name__ == "__main__":
    unittest.main()
