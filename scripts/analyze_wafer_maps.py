"""Run wafer map spatial defect analysis."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.wafer_ai_outputs import run_optional_wafer_ai_outputs
from src.wafer_map_analysis import (
    WaferMapRecord,
    load_wafer_map_records,
    run_wafer_map_analysis,
)


def parse_optional_text(value: str | None) -> str | None:
    """Parse optional text arguments."""
    if value is None or not str(value).strip():
        return None
    return value


def parse_coordinate_defect_value(value: str) -> int | float | str:
    """Parse coordinate CSV defect values while preserving non-numeric labels."""
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def parse_positive_int(value: str) -> int:
    """Parse a strictly positive integer CLI value."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than 0")
    return parsed


def parse_non_negative_int(value: str) -> int:
    """Parse a non-negative integer CLI value."""
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be 0 or greater")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze spatial defect patterns from wafer maps.")
    parser.add_argument("--input-path", type=Path, default=None, help="WM-811K pickle, .npy/.npz, or coordinate CSV.")
    parser.add_argument(
        "--input-format",
        choices=["auto", "wm811k", "array", "coordinate"],
        default="auto",
        help="Input format. 'auto' infers from file extension.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/wafer_maps"))
    parser.add_argument("--demo", action="store_true", help="Run with synthetic demo wafer maps.")
    parser.add_argument("--defect-value", type=float, default=2.0, help="Defective die value for array/WM-811K maps.")
    parser.add_argument("--n-clusters", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-images", type=int, default=12)
    parser.add_argument(
        "--similarity-max-records",
        type=parse_non_negative_int,
        default=5000,
        help="Limit wafer maps used for pairwise similarity search. Use 0 to skip similarity search.",
    )
    parser.add_argument(
        "--max-records",
        type=parse_positive_int,
        default=None,
        help="Limit wafer maps analyzed to a positive count. Omit to analyze all records.",
    )

    parser.add_argument("--wafer-map-col", default="waferMap")
    parser.add_argument("--label-col", type=parse_optional_text, default="failureType")
    parser.add_argument("--id-col", type=parse_optional_text, default=None)
    parser.add_argument("--group-col", type=parse_optional_text, default="lotName")
    parser.add_argument("--wafer-id-col", default="wafer_id")
    parser.add_argument("--x-col", default="x")
    parser.add_argument("--y-col", default="y")
    parser.add_argument("--value-col", type=parse_optional_text, default=None)
    parser.add_argument("--defect-col", type=parse_optional_text, default=None)
    parser.add_argument("--coordinate-defect-value", type=parse_coordinate_defect_value, default=1)

    parser.add_argument("--train-cnn", action="store_true", help="Train a small CNN classifier when labels exist.")
    parser.add_argument("--cnn-epochs", type=int, default=5)
    parser.add_argument("--autoencoder", action="store_true", help="Train autoencoder anomaly scoring model.")
    parser.add_argument("--autoencoder-epochs", type=int, default=5)
    parser.add_argument("--ai-backend", choices=["pytorch", "tensorflow"], default="pytorch")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--ai-resize-to", type=parse_positive_int, default=32)
    parser.add_argument("--ai-batch-size", type=parse_positive_int, default=32)
    parser.add_argument(
        "--export-onnx",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Export PyTorch models to ONNX when the ONNX package is installed.",
    )
    return parser.parse_args()


def make_demo_records(
    size: int = 33,
    variants_per_pattern: int = 1,
) -> list[WaferMapRecord]:
    """Create small synthetic wafer maps for smoke testing the workflow."""
    if variants_per_pattern <= 0:
        raise ValueError("variants_per_pattern must be greater than 0.")
    rows, cols = np.indices((size, size))
    center = (size - 1) / 2
    radius = np.sqrt((rows - center) ** 2 + (cols - center) ** 2)
    valid = radius <= center

    def blank() -> np.ndarray:
        wafer_map = np.zeros((size, size), dtype=int)
        wafer_map[valid] = 1
        return wafer_map

    records: list[WaferMapRecord] = []

    center_map = blank()
    center_map[radius <= size * 0.12] = 2
    records.append(WaferMapRecord("demo_center", center_map, "Center"))

    edge_ring = blank()
    edge_ring[(radius >= size * 0.38) & valid] = 2
    records.append(WaferMapRecord("demo_edge_ring", edge_ring, "Edge-Ring"))

    edge_local = blank()
    edge_local[(cols > size * 0.70) & (rows > size * 0.35) & (rows < size * 0.65) & valid] = 2
    records.append(WaferMapRecord("demo_edge_local", edge_local, "Edge-Local"))

    scratch = blank()
    for offset in range(-1, 2):
        scratch[np.clip(np.arange(6, 27), 0, size - 1), np.clip(np.arange(7, 28) + offset, 0, size - 1)] = 2
    scratch[~valid] = 0
    records.append(WaferMapRecord("demo_scratch", scratch, "Scratch"))

    donut = blank()
    donut[(radius > size * 0.16) & (radius < size * 0.25)] = 2
    records.append(WaferMapRecord("demo_donut", donut, "Donut"))

    near_full = blank()
    near_full[valid & (rows + cols > size * 0.45)] = 2
    records.append(WaferMapRecord("demo_near_full", near_full, "Near-Full"))

    grouped_records = [
        WaferMapRecord(
            wafer_id=record.wafer_id,
            wafer_map=record.wafer_map,
            label=record.label,
            group_id="demo_lot_0",
        )
        for record in records
    ]
    for variant in range(1, variants_per_pattern):
        for record in records:
            valid_mask = record.wafer_map > 0
            defect_mask = record.wafer_map == 2
            shifted = np.roll(defect_mask, shift=(variant, -variant), axis=(0, 1))
            variant_map = np.where(valid_mask, 1, 0)
            variant_map[shifted & valid_mask] = 2
            grouped_records.append(
                WaferMapRecord(
                    wafer_id=f"{record.wafer_id}_v{variant}",
                    wafer_map=variant_map,
                    label=record.label,
                    group_id=f"demo_lot_{variant}",
                )
            )
    return grouped_records


def load_records_from_args(args: argparse.Namespace) -> list[WaferMapRecord]:
    """Load user-provided or demo wafer map records."""
    if args.demo:
        ai_requested = bool(getattr(args, "train_cnn", False) or getattr(args, "autoencoder", False))
        records = make_demo_records(variants_per_pattern=3 if ai_requested else 1)
        return records[: args.max_records] if args.max_records is not None else records
    if args.input_path is None:
        raise ValueError("Pass --input-path or use --demo.")
    records = load_wafer_map_records(
        path=args.input_path,
        input_format=args.input_format,
        wafer_map_col=args.wafer_map_col,
        label_col=args.label_col,
        id_col=args.id_col,
        group_col=args.group_col,
        wafer_id_col=args.wafer_id_col,
        x_col=args.x_col,
        y_col=args.y_col,
        value_col=args.value_col,
        defect_col=args.defect_col,
        coordinate_defect_value=args.coordinate_defect_value,
    )
    return records[: args.max_records] if args.max_records is not None else records


def run_optional_ai_outputs(
    records: list[WaferMapRecord],
    output_dir: Path,
    train_cnn: bool = False,
    cnn_epochs: int = 5,
    autoencoder: bool = False,
    autoencoder_epochs: int = 5,
    backend: str = "pytorch",
    device: str = "auto",
    resize_to: int = 32,
    batch_size: int = 32,
    export_onnx: bool = True,
) -> dict[str, Path]:
    """Run optional wafer AI outputs with a selectable training backend."""
    return run_optional_wafer_ai_outputs(
        records=records,
        output_dir=output_dir,
        train_cnn=train_cnn,
        cnn_epochs=cnn_epochs,
        autoencoder=autoencoder,
        autoencoder_epochs=autoencoder_epochs,
        backend=backend,
        device=device,
        resize_to=resize_to,
        batch_size=batch_size,
        export_onnx=export_onnx,
    )


def analyze_wafer_maps_from_args(args: argparse.Namespace) -> dict[str, Path]:
    """Run the CLI workflow and return output paths."""
    records = load_records_from_args(args)
    outputs = run_wafer_map_analysis(
        records=records,
        output_dir=args.output_dir,
        defect_value=args.defect_value,
        n_clusters=args.n_clusters,
        top_k=args.top_k,
        max_images=args.max_images,
        similarity_max_records=args.similarity_max_records,
    )
    outputs.update(
        run_optional_ai_outputs(
            records=records,
            output_dir=args.output_dir,
            train_cnn=args.train_cnn,
            cnn_epochs=args.cnn_epochs,
            autoencoder=args.autoencoder,
            autoencoder_epochs=args.autoencoder_epochs,
            backend=args.ai_backend,
            device=args.device,
            resize_to=args.ai_resize_to,
            batch_size=args.ai_batch_size,
            export_onnx=args.export_onnx,
        )
    )
    return outputs


def main() -> int:
    args = parse_args()
    outputs = analyze_wafer_maps_from_args(args)
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
