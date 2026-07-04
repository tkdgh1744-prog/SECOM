"""Run wafer map spatial defect analysis."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.wafer_map_analysis import (
    WaferMapRecord,
    load_wafer_map_records,
    run_wafer_map_analysis,
    train_autoencoder_anomaly_scores,
    train_cnn_pattern_classifier,
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

    parser.add_argument("--wafer-map-col", default="waferMap")
    parser.add_argument("--label-col", type=parse_optional_text, default="failureType")
    parser.add_argument("--id-col", type=parse_optional_text, default=None)
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
    return parser.parse_args()


def make_demo_records(size: int = 33) -> list[WaferMapRecord]:
    """Create small synthetic wafer maps for smoke testing the workflow."""
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

    return records


def load_records_from_args(args: argparse.Namespace) -> list[WaferMapRecord]:
    """Load user-provided or demo wafer map records."""
    if args.demo:
        return make_demo_records()
    if args.input_path is None:
        raise ValueError("Pass --input-path or use --demo.")
    return load_wafer_map_records(
        path=args.input_path,
        input_format=args.input_format,
        wafer_map_col=args.wafer_map_col,
        label_col=args.label_col,
        id_col=args.id_col,
        wafer_id_col=args.wafer_id_col,
        x_col=args.x_col,
        y_col=args.y_col,
        value_col=args.value_col,
        defect_col=args.defect_col,
        coordinate_defect_value=args.coordinate_defect_value,
    )


def run_optional_ai_outputs(
    records: list[WaferMapRecord],
    output_dir: Path,
    train_cnn: bool = False,
    cnn_epochs: int = 5,
    autoencoder: bool = False,
    autoencoder_epochs: int = 5,
) -> dict[str, Path]:
    """Run optional TensorFlow-based AI outputs."""
    outputs: dict[str, Path] = {}
    output_dir.mkdir(parents=True, exist_ok=True)

    if train_cnn:
        labeled_records = [record for record in records if record.label]
        labels = [str(record.label) for record in labeled_records]
        if len(labeled_records) < 4 or len(set(labels)) < 2:
            skip_path = output_dir / "cnn_skipped.txt"
            skip_path.write_text("CNN skipped: at least 4 labeled maps across 2 classes are required.\n", encoding="utf-8")
            outputs["cnn_skipped"] = skip_path
        else:
            try:
                cnn_result = train_cnn_pattern_classifier(labeled_records, labels, epochs=cnn_epochs)
            except ImportError as exc:
                skip_path = output_dir / "cnn_skipped.txt"
                skip_path.write_text(f"CNN skipped: {exc}\n", encoding="utf-8")
                outputs["cnn_skipped"] = skip_path
            else:
                model_path = output_dir / "cnn_pattern_classifier.keras"
                metrics_path = output_dir / "cnn_metrics.csv"
                classes_path = output_dir / "cnn_label_classes.csv"
                cnn_result["model"].save(model_path)
                pd.DataFrame(
                    [{"test_loss": cnn_result["test_loss"], "test_accuracy": cnn_result["test_accuracy"]}]
                ).to_csv(metrics_path, index=False)
                pd.DataFrame({"class_name": cnn_result["classes"]}).to_csv(classes_path, index=False)
                outputs["cnn_model"] = model_path
                outputs["cnn_metrics"] = metrics_path
                outputs["cnn_classes"] = classes_path

    if autoencoder:
        try:
            scores = train_autoencoder_anomaly_scores(records, epochs=autoencoder_epochs)
        except ImportError as exc:
            skip_path = output_dir / "autoencoder_skipped.txt"
            skip_path.write_text(f"Autoencoder skipped: {exc}\n", encoding="utf-8")
            outputs["autoencoder_skipped"] = skip_path
        else:
            scores_path = output_dir / "autoencoder_anomaly_scores.csv"
            scores.to_csv(scores_path, index=False)
            outputs["autoencoder_scores"] = scores_path

    return outputs


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
    )
    outputs.update(
        run_optional_ai_outputs(
            records=records,
            output_dir=args.output_dir,
            train_cnn=args.train_cnn,
            cnn_epochs=args.cnn_epochs,
            autoencoder=args.autoencoder,
            autoencoder_epochs=args.autoencoder_epochs,
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

