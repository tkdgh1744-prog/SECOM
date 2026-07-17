from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.analyze_wafer_maps import make_demo_records
from src.model_profiling import (
    profile_onnx_model,
    profile_pytorch_bundle,
    quantize_onnx_dynamic_int8,
    summarize_latencies,
    write_profile_outputs,
)
from src.wafer_torch import (
    export_torch_model_to_onnx,
    onnx_is_available,
    onnxruntime_is_available,
    save_torch_model_bundle,
    torch_is_available,
    train_torch_cnn_pattern_classifier,
)


class ModelProfilingUtilityTests(unittest.TestCase):
    def test_latency_summary_reports_percentiles(self) -> None:
        summary = summarize_latencies([1.0, 2.0, 3.0, 4.0])

        self.assertEqual(summary["latency_mean_ms"], 2.5)
        self.assertEqual(summary["latency_p50_ms"], 2.5)
        self.assertGreater(summary["latency_p95_ms"], 3.0)

    def test_latency_summary_rejects_invalid_values(self) -> None:
        with self.assertRaises(ValueError):
            summarize_latencies([])
        with self.assertRaises(ValueError):
            summarize_latencies([1.0, -1.0])

    def test_profile_outputs_are_machine_readable(self) -> None:
        profile = {
            "model_name": "demo",
            "runtime": "onnxruntime",
            "precision": "FP32",
            "input_shape": [1, 1, 32, 32],
            "operator_counts": {"Conv": 2, "Relu": 2},
            "latency_mean_ms": 1.25,
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            outputs = write_profile_outputs([profile], Path(tmpdir))

            rows = pd.read_csv(outputs["profiles_csv"])
            operations = pd.read_csv(outputs["operations_csv"])
            report = json.loads(outputs["report_json"].read_text(encoding="utf-8"))
            self.assertEqual(rows.loc[0, "model_name"], "demo")
            self.assertEqual(set(operations["operator"]), {"Conv", "Relu"})
            self.assertEqual(report["profiles"][0]["input_shape"], [1, 1, 32, 32])


@unittest.skipUnless(
    torch_is_available() and onnx_is_available() and onnxruntime_is_available(),
    "PyTorch, ONNX, or ONNX Runtime is not installed",
)
class ModelProfilingRuntimeTests(unittest.TestCase):
    def test_profiles_pytorch_fp32_onnx_and_int8(self) -> None:
        records = make_demo_records(variants_per_pattern=3)
        result = train_torch_cnn_pattern_classifier(
            records,
            [str(record.label) for record in records],
            epochs=1,
            batch_size=18,
            device="cpu",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            pytorch_path = save_torch_model_bundle(
                result,
                output_dir / "wafer_cnn.pt",
                model_type="wafer_pattern_cnn",
            )
            onnx_path = export_torch_model_to_onnx(
                result,
                output_dir / "wafer_cnn.onnx",
            )
            int8_path = quantize_onnx_dynamic_int8(
                onnx_path,
                output_dir / "wafer_cnn.int8.onnx",
            )

            pytorch_profile = profile_pytorch_bundle(
                pytorch_path,
                warmup_runs=1,
                measured_runs=2,
            )
            onnx_profile = profile_onnx_model(
                onnx_path,
                warmup_runs=1,
                measured_runs=2,
            )
            int8_profile = profile_onnx_model(
                int8_path,
                warmup_runs=1,
                measured_runs=2,
                precision="INT8",
            )

            self.assertEqual(pytorch_profile["runtime"], "pytorch")
            self.assertEqual(onnx_profile["provider"], "CPUExecutionProvider")
            self.assertEqual(int8_profile["precision"], "INT8")
            self.assertGreater(onnx_profile["operation_node_count"], 0)
            self.assertLess(int8_profile["model_size_bytes"], onnx_profile["model_size_bytes"])
            self.assertGreater(pytorch_profile["throughput_items_per_second"], 0)


if __name__ == "__main__":
    unittest.main()
