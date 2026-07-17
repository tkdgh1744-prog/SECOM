"""Profile PyTorch and ONNX wafer models on the current CPU runtime."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model_profiling import (
    profile_onnx_model,
    profile_pytorch_bundle,
    quantize_onnx_dynamic_int8,
    write_profile_outputs,
)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than 0")
    return parsed


def non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be 0 or greater")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure wafer model size, memory proxies, operations, and latency."
    )
    parser.add_argument("--pytorch-model", type=Path, default=None)
    parser.add_argument("--onnx-model", type=Path, default=None)
    parser.add_argument(
        "--quantize-int8",
        action="store_true",
        help="Create and profile a dynamically weight-quantized INT8 ONNX model.",
    )
    parser.add_argument("--int8-output", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/profiling"))
    parser.add_argument("--batch-size", type=positive_int, default=1)
    parser.add_argument("--warmup-runs", type=non_negative_int, default=10)
    parser.add_argument("--measured-runs", type=positive_int, default=50)
    parser.add_argument("--input-size", type=positive_int, default=None)
    parser.add_argument("--intra-op-threads", type=positive_int, default=1)
    return parser.parse_args()


def profile_from_args(args: argparse.Namespace) -> dict[str, Path]:
    if args.pytorch_model is None and args.onnx_model is None:
        raise ValueError("Pass --pytorch-model, --onnx-model, or both.")
    if args.quantize_int8 and args.onnx_model is None:
        raise ValueError("--quantize-int8 requires --onnx-model.")

    profiles: list[dict[str, object]] = []
    if args.pytorch_model is not None:
        profiles.append(
            profile_pytorch_bundle(
                args.pytorch_model,
                batch_size=args.batch_size,
                warmup_runs=args.warmup_runs,
                measured_runs=args.measured_runs,
                device="cpu",
            )
        )
    if args.onnx_model is not None:
        profiles.append(
            profile_onnx_model(
                args.onnx_model,
                batch_size=args.batch_size,
                warmup_runs=args.warmup_runs,
                measured_runs=args.measured_runs,
                input_size=args.input_size,
                intra_op_threads=args.intra_op_threads,
                precision="FP32",
            )
        )
    outputs: dict[str, Path] = {}
    if args.quantize_int8:
        int8_path = args.int8_output or (
            args.output_dir / f"{args.onnx_model.stem}.int8.onnx"
        )
        outputs["int8_model"] = quantize_onnx_dynamic_int8(args.onnx_model, int8_path)
        profiles.append(
            profile_onnx_model(
                int8_path,
                batch_size=args.batch_size,
                warmup_runs=args.warmup_runs,
                measured_runs=args.measured_runs,
                input_size=args.input_size,
                intra_op_threads=args.intra_op_threads,
                precision="INT8",
            )
        )
    outputs.update(write_profile_outputs(profiles, args.output_dir))
    return outputs


def main() -> int:
    args = parse_args()
    outputs = profile_from_args(args)
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
