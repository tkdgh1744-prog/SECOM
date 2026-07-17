"""Persist optional wafer AI outputs for PyTorch and legacy TensorFlow backends."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.wafer_map_analysis import (
    WaferMapRecord,
    train_autoencoder_anomaly_scores,
    train_cnn_pattern_classifier,
)
from src.wafer_torch import (
    export_torch_model_to_onnx,
    save_torch_model_bundle,
    train_torch_autoencoder_anomaly_scores,
    train_torch_cnn_pattern_classifier,
)


def _write_skip(output_dir: Path, name: str, message: str) -> Path:
    path = output_dir / name
    path.write_text(message.rstrip() + "\n", encoding="utf-8")
    return path


def _export_onnx_or_skip(
    result: dict[str, object],
    output_dir: Path,
    stem: str,
    outputs: dict[str, Path],
) -> None:
    try:
        onnx_path = export_torch_model_to_onnx(result, output_dir / f"{stem}.onnx")
    except ImportError as exc:
        outputs[f"{stem}_onnx_skipped"] = _write_skip(
            output_dir,
            f"{stem}_onnx_skipped.txt",
            f"ONNX export skipped: {exc}",
        )
    else:
        outputs[f"{stem}_onnx"] = onnx_path


def _run_pytorch_cnn(
    records: list[WaferMapRecord],
    labels: list[str],
    output_dir: Path,
    epochs: int,
    resize_to: int,
    batch_size: int,
    device: str,
    export_onnx: bool,
) -> dict[str, Path]:
    outputs: dict[str, Path] = {}
    try:
        result = train_torch_cnn_pattern_classifier(
            records,
            labels,
            epochs=epochs,
            resize_to=resize_to,
            batch_size=batch_size,
            device=device,
        )
    except ImportError as exc:
        outputs["cnn_skipped"] = _write_skip(
            output_dir,
            "cnn_skipped.txt",
            f"CNN skipped: {exc}",
        )
        return outputs

    model_path = save_torch_model_bundle(
        result,
        output_dir / "cnn_pattern_classifier.pt",
        model_type="wafer_pattern_cnn",
    )
    metrics_path = output_dir / "cnn_metrics.csv"
    history_path = output_dir / "cnn_training_history.csv"
    classes_path = output_dir / "cnn_label_classes.csv"
    split_path = output_dir / "cnn_grouped_split.csv"
    pd.DataFrame(
        [
            {
                "framework": "pytorch",
                "device": result["device"],
                "parameter_count": result["parameter_count"],
                "test_loss": result["test_loss"],
                "test_accuracy": result["test_accuracy"],
            }
        ]
    ).to_csv(metrics_path, index=False)
    result["history"].to_csv(history_path, index=False)
    pd.DataFrame({"class_name": result["classes"]}).to_csv(classes_path, index=False)
    result["split"].to_csv(split_path, index=False)
    outputs.update(
        {
            "cnn_model": model_path,
            "cnn_metrics": metrics_path,
            "cnn_history": history_path,
            "cnn_classes": classes_path,
            "cnn_split": split_path,
        }
    )
    if export_onnx:
        _export_onnx_or_skip(result, output_dir, "cnn_pattern_classifier", outputs)
    return outputs


def _run_tensorflow_cnn(
    records: list[WaferMapRecord],
    labels: list[str],
    output_dir: Path,
    epochs: int,
    resize_to: int,
) -> dict[str, Path]:
    outputs: dict[str, Path] = {}
    try:
        result = train_cnn_pattern_classifier(
            records,
            labels,
            epochs=epochs,
            resize_to=resize_to,
        )
    except ImportError as exc:
        outputs["cnn_skipped"] = _write_skip(
            output_dir,
            "cnn_skipped.txt",
            f"CNN skipped: {exc}",
        )
        return outputs

    model_path = output_dir / "cnn_pattern_classifier.keras"
    metrics_path = output_dir / "cnn_metrics.csv"
    classes_path = output_dir / "cnn_label_classes.csv"
    result["model"].save(model_path)
    pd.DataFrame(
        [
            {
                "framework": "tensorflow",
                "test_loss": result["test_loss"],
                "test_accuracy": result["test_accuracy"],
            }
        ]
    ).to_csv(metrics_path, index=False)
    pd.DataFrame({"class_name": result["classes"]}).to_csv(classes_path, index=False)
    outputs.update(
        {
            "cnn_model": model_path,
            "cnn_metrics": metrics_path,
            "cnn_classes": classes_path,
        }
    )
    return outputs


def _run_pytorch_autoencoder(
    records: list[WaferMapRecord],
    output_dir: Path,
    epochs: int,
    resize_to: int,
    batch_size: int,
    device: str,
    export_onnx: bool,
) -> dict[str, Path]:
    outputs: dict[str, Path] = {}
    try:
        result = train_torch_autoencoder_anomaly_scores(
            records,
            epochs=epochs,
            resize_to=resize_to,
            batch_size=batch_size,
            device=device,
        )
    except ImportError as exc:
        outputs["autoencoder_skipped"] = _write_skip(
            output_dir,
            "autoencoder_skipped.txt",
            f"Autoencoder skipped: {exc}",
        )
        return outputs

    model_path = save_torch_model_bundle(
        result,
        output_dir / "wafer_autoencoder.pt",
        model_type="wafer_autoencoder",
    )
    scores_path = output_dir / "autoencoder_anomaly_scores.csv"
    history_path = output_dir / "autoencoder_training_history.csv"
    metrics_path = output_dir / "autoencoder_metrics.csv"
    result["scores"].to_csv(scores_path, index=False)
    result["history"].to_csv(history_path, index=False)
    pd.DataFrame(
        [
            {
                "framework": "pytorch",
                "device": result["device"],
                "parameter_count": result["parameter_count"],
            }
        ]
    ).to_csv(metrics_path, index=False)
    outputs.update(
        {
            "autoencoder_model": model_path,
            "autoencoder_scores": scores_path,
            "autoencoder_history": history_path,
            "autoencoder_metrics": metrics_path,
        }
    )
    if export_onnx:
        _export_onnx_or_skip(result, output_dir, "wafer_autoencoder", outputs)
    return outputs


def _run_tensorflow_autoencoder(
    records: list[WaferMapRecord],
    output_dir: Path,
    epochs: int,
    resize_to: int,
) -> dict[str, Path]:
    outputs: dict[str, Path] = {}
    try:
        scores = train_autoencoder_anomaly_scores(
            records,
            epochs=epochs,
            resize_to=resize_to,
        )
    except ImportError as exc:
        outputs["autoencoder_skipped"] = _write_skip(
            output_dir,
            "autoencoder_skipped.txt",
            f"Autoencoder skipped: {exc}",
        )
        return outputs
    scores_path = output_dir / "autoencoder_anomaly_scores.csv"
    scores.to_csv(scores_path, index=False)
    outputs["autoencoder_scores"] = scores_path
    return outputs


def run_optional_wafer_ai_outputs(
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
    """Train requested wafer models and persist backend-specific artifacts."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    normalized_backend = str(backend).strip().lower()
    if normalized_backend not in {"pytorch", "tensorflow"}:
        raise ValueError("backend must be pytorch or tensorflow")

    outputs: dict[str, Path] = {}
    if train_cnn:
        labeled_records = [record for record in records if record.label]
        labels = [str(record.label) for record in labeled_records]
        if len(labeled_records) < 4 or len(set(labels)) < 2:
            outputs["cnn_skipped"] = _write_skip(
                output_dir,
                "cnn_skipped.txt",
                "CNN skipped: at least 4 labeled maps across 2 classes are required.",
            )
        elif normalized_backend == "pytorch":
            outputs.update(
                _run_pytorch_cnn(
                    labeled_records,
                    labels,
                    output_dir,
                    cnn_epochs,
                    resize_to,
                    batch_size,
                    device,
                    export_onnx,
                )
            )
        else:
            outputs.update(
                _run_tensorflow_cnn(
                    labeled_records,
                    labels,
                    output_dir,
                    cnn_epochs,
                    resize_to,
                )
            )

    if autoencoder:
        if normalized_backend == "pytorch":
            outputs.update(
                _run_pytorch_autoencoder(
                    records,
                    output_dir,
                    autoencoder_epochs,
                    resize_to,
                    batch_size,
                    device,
                    export_onnx,
                )
            )
        else:
            outputs.update(
                _run_tensorflow_autoencoder(
                    records,
                    output_dir,
                    autoencoder_epochs,
                    resize_to,
                )
            )
    return outputs
