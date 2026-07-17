"""Optional PyTorch training and ONNX export for wafer-map models."""

from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.wafer_map_analysis import WaferMapRecord, infer_wafer_masks, resize_binary_map


def torch_is_available() -> bool:
    """Return whether PyTorch can be imported in the current runtime."""
    return find_spec("torch") is not None


def onnx_is_available() -> bool:
    """Return whether the ONNX package required by the exporter is installed."""
    return find_spec("onnx") is not None


def onnxruntime_is_available() -> bool:
    """Return whether ONNX Runtime can execute exported models."""
    return find_spec("onnxruntime") is not None


def _require_torch() -> tuple[Any, Any]:
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required for wafer AI training. Install requirements-ai.txt."
        ) from exc
    return torch, nn


def resolve_torch_device(requested: str = "auto") -> str:
    """Resolve auto/cpu/cuda/mps to an available PyTorch device."""
    torch, _ = _require_torch()
    normalized = str(requested).strip().lower()
    if normalized == "auto":
        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
        return "cpu"
    if normalized == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available.")
    if normalized == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            raise ValueError("MPS was requested but is not available.")
    if normalized not in {"cpu", "cuda", "mps"}:
        raise ValueError("device must be one of: auto, cpu, cuda, mps")
    return normalized


def prepare_wafer_images(
    records: list[WaferMapRecord],
    resize_to: int = 32,
) -> np.ndarray:
    """Convert wafer maps to normalized NCHW defect-mask tensors."""
    if not records:
        raise ValueError("At least one wafer map record is required.")
    if resize_to <= 0:
        raise ValueError("resize_to must be greater than 0.")
    images = []
    for record in records:
        _, defect_mask = infer_wafer_masks(record.wafer_map)
        images.append(resize_binary_map(defect_mask, size=resize_to))
    return np.asarray(images, dtype=np.float32)[:, np.newaxis, :, :]


def grouped_train_test_indices(
    records: list[WaferMapRecord],
    labels: list[str] | None = None,
    test_fraction: float = 0.2,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Split whole wafer groups so a lot/wafer group never crosses the boundary."""
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1.")
    if labels is not None and len(labels) != len(records):
        raise ValueError("labels must have the same length as records.")

    groups = np.asarray([record.group_id or record.wafer_id for record in records], dtype=object)
    unique_groups = np.asarray(sorted(set(groups.tolist())), dtype=object)
    if len(unique_groups) < 2:
        raise ValueError("At least two distinct wafer groups are required for evaluation.")

    test_group_count = min(
        len(unique_groups) - 1,
        max(1, int(round(len(unique_groups) * test_fraction))),
    )
    rng = np.random.default_rng(random_state)
    required_train_labels = set(labels or [])
    attempts = max(32, len(unique_groups) * 4)

    for _ in range(attempts):
        shuffled = rng.permutation(unique_groups)
        test_groups = set(shuffled[:test_group_count].tolist())
        test_mask = np.asarray([group in test_groups for group in groups], dtype=bool)
        train_indices = np.flatnonzero(~test_mask)
        test_indices = np.flatnonzero(test_mask)
        if not len(train_indices) or not len(test_indices):
            continue
        if labels is not None:
            train_labels = {labels[index] for index in train_indices}
            if not required_train_labels.issubset(train_labels):
                continue
        return train_indices, test_indices

    raise ValueError(
        "Could not create a grouped split containing every class in training. "
        "Add more lots/groups for rare classes."
    )


def _seed_torch(torch: Any, random_state: int) -> None:
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_state)


def _build_classifier(nn: Any, class_count: int) -> Any:
    class WaferPatternCNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((4, 4)),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(32 * 4 * 4, 64),
                nn.ReLU(),
                nn.Linear(64, class_count),
            )

        def forward(self, inputs: Any) -> Any:
            return self.classifier(self.features(inputs))

    return WaferPatternCNN()


def _build_autoencoder(nn: Any) -> Any:
    class WaferAutoencoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.network = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 8, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.ConvTranspose2d(8, 16, kernel_size=2, stride=2),
                nn.ReLU(),
                nn.ConvTranspose2d(16, 1, kernel_size=2, stride=2),
                nn.Sigmoid(),
            )

        def forward(self, inputs: Any) -> Any:
            return self.network(inputs)

    return WaferAutoencoder()


def _classification_metrics(
    torch: Any,
    model: Any,
    loader: Any,
    loss_function: Any,
    device: str,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    seen = 0
    with torch.no_grad():
        for batch_inputs, batch_targets in loader:
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)
            logits = model(batch_inputs)
            total_loss += float(loss_function(logits, batch_targets).item()) * len(batch_inputs)
            correct += int((logits.argmax(dim=1) == batch_targets).sum().item())
            seen += len(batch_inputs)
    return total_loss / max(seen, 1), correct / max(seen, 1)


def train_torch_cnn_pattern_classifier(
    records: list[WaferMapRecord],
    labels: list[str],
    epochs: int = 5,
    resize_to: int = 32,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    test_fraction: float = 0.2,
    random_state: int = 42,
    device: str = "auto",
) -> dict[str, object]:
    """Train a compact PyTorch CNN with a leakage-resistant grouped split."""
    torch, nn = _require_torch()
    if len(records) != len(labels):
        raise ValueError("records and labels must have the same length.")
    if epochs <= 0 or batch_size <= 0:
        raise ValueError("epochs and batch_size must be greater than 0.")
    classes = sorted(set(str(label) for label in labels))
    if len(classes) < 2:
        raise ValueError("At least two pattern classes are required.")

    normalized_labels = [str(label) for label in labels]
    class_to_index = {label: index for index, label in enumerate(classes)}
    encoded = np.asarray([class_to_index[label] for label in normalized_labels], dtype=np.int64)
    images = prepare_wafer_images(records, resize_to=resize_to)
    train_indices, test_indices = grouped_train_test_indices(
        records,
        labels=normalized_labels,
        test_fraction=test_fraction,
        random_state=random_state,
    )
    resolved_device = resolve_torch_device(device)
    _seed_torch(torch, random_state)

    model = _build_classifier(nn, len(classes)).to(resolved_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.CrossEntropyLoss()
    train_inputs = torch.from_numpy(images[train_indices])
    train_targets = torch.from_numpy(encoded[train_indices])
    test_inputs = torch.from_numpy(images[test_indices])
    test_targets = torch.from_numpy(encoded[test_indices])
    generator = torch.Generator().manual_seed(random_state)
    train_dataset = torch.utils.data.TensorDataset(train_inputs, train_targets)
    test_dataset = torch.utils.data.TensorDataset(test_inputs, test_targets)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=min(batch_size, len(train_dataset)),
        shuffle=True,
        generator=generator,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=min(batch_size, len(test_dataset)),
        shuffle=False,
    )

    history: list[dict[str, float | int]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        seen = 0
        for batch_inputs, batch_targets in train_loader:
            batch_inputs = batch_inputs.to(resolved_device)
            batch_targets = batch_targets.to(resolved_device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_inputs)
            loss = loss_function(logits, batch_targets)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(batch_inputs)
            seen += len(batch_inputs)
        test_loss, test_accuracy = _classification_metrics(
            torch,
            model,
            test_loader,
            loss_function,
            resolved_device,
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / max(seen, 1),
                "test_loss": test_loss,
                "test_accuracy": test_accuracy,
            }
        )

    test_set = set(test_indices.tolist())
    split = pd.DataFrame(
        {
            "wafer_id": [record.wafer_id for record in records],
            "group_id": [record.group_id or record.wafer_id for record in records],
            "label": normalized_labels,
            "split": ["test" if index in test_set else "train" for index in range(len(records))],
        }
    )
    return {
        "model": model,
        "classes": classes,
        "history": pd.DataFrame(history),
        "split": split,
        "test_loss": history[-1]["test_loss"],
        "test_accuracy": history[-1]["test_accuracy"],
        "device": resolved_device,
        "input_size": resize_to,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }


def train_torch_autoencoder_anomaly_scores(
    records: list[WaferMapRecord],
    epochs: int = 5,
    resize_to: int = 32,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    test_fraction: float = 0.2,
    random_state: int = 42,
    device: str = "auto",
) -> dict[str, object]:
    """Train an autoencoder on training groups and score every wafer map."""
    torch, nn = _require_torch()
    if resize_to % 4:
        raise ValueError("resize_to must be divisible by 4 for the autoencoder.")
    if epochs <= 0 or batch_size <= 0:
        raise ValueError("epochs and batch_size must be greater than 0.")
    images = prepare_wafer_images(records, resize_to=resize_to)
    train_indices, test_indices = grouped_train_test_indices(
        records,
        test_fraction=test_fraction,
        random_state=random_state,
    )
    resolved_device = resolve_torch_device(device)
    _seed_torch(torch, random_state)

    model = _build_autoencoder(nn).to(resolved_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.MSELoss()
    train_inputs = torch.from_numpy(images[train_indices])
    generator = torch.Generator().manual_seed(random_state)
    train_dataset = torch.utils.data.TensorDataset(train_inputs, train_inputs)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=min(batch_size, len(train_dataset)),
        shuffle=True,
        generator=generator,
    )

    history: list[dict[str, float | int]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        seen = 0
        for batch_inputs, batch_targets in train_loader:
            batch_inputs = batch_inputs.to(resolved_device)
            batch_targets = batch_targets.to(resolved_device)
            optimizer.zero_grad(set_to_none=True)
            reconstruction = model(batch_inputs)
            loss = loss_function(reconstruction, batch_targets)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(batch_inputs)
            seen += len(batch_inputs)
        history.append({"epoch": epoch, "train_loss": total_loss / max(seen, 1)})

    score_dataset = torch.utils.data.TensorDataset(torch.from_numpy(images))
    score_loader = torch.utils.data.DataLoader(
        score_dataset,
        batch_size=min(batch_size, len(score_dataset)),
        shuffle=False,
    )
    error_batches = []
    model.eval()
    with torch.no_grad():
        for (batch_inputs,) in score_loader:
            batch_inputs = batch_inputs.to(resolved_device)
            reconstruction = model(batch_inputs)
            batch_errors = ((batch_inputs - reconstruction) ** 2).mean(dim=(1, 2, 3))
            error_batches.append(batch_errors.cpu().numpy())
    errors = np.concatenate(error_batches)
    test_set = set(test_indices.tolist())
    scores = pd.DataFrame(
        {
            "wafer_id": [record.wafer_id for record in records],
            "group_id": [record.group_id or record.wafer_id for record in records],
            "split": ["test" if index in test_set else "train" for index in range(len(records))],
            "autoencoder_reconstruction_error": errors.astype(float),
        }
    ).sort_values("autoencoder_reconstruction_error", ascending=False)
    return {
        "model": model,
        "history": pd.DataFrame(history),
        "scores": scores,
        "device": resolved_device,
        "input_size": resize_to,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }


def save_torch_model_bundle(
    result: dict[str, object],
    output_path: Path,
    model_type: str,
) -> Path:
    """Save model weights with portable reconstruction metadata."""
    torch, _ = _require_torch()
    model = result["model"]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_type": model_type,
        "state_dict": {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()},
        "classes": list(result.get("classes", [])),
        "input_size": int(result["input_size"]),
        "parameter_count": int(result["parameter_count"]),
        "framework": "pytorch",
    }
    torch.save(payload, output_path)
    return output_path


def load_torch_model_bundle(
    model_path: Path,
    device: str = "cpu",
) -> dict[str, object]:
    """Load a trusted state-dict bundle and reconstruct its model architecture."""
    torch, nn = _require_torch()
    resolved_device = resolve_torch_device(device)
    payload = torch.load(
        Path(model_path),
        map_location=resolved_device,
        weights_only=True,
    )
    if not isinstance(payload, dict):
        raise ValueError("Invalid PyTorch wafer model bundle.")
    model_type = str(payload.get("model_type", ""))
    classes = [str(value) for value in payload.get("classes", [])]
    if model_type == "wafer_pattern_cnn":
        if len(classes) < 2:
            raise ValueError("CNN bundle must contain at least two classes.")
        model = _build_classifier(nn, len(classes))
    elif model_type == "wafer_autoencoder":
        model = _build_autoencoder(nn)
    else:
        raise ValueError(f"Unsupported wafer model type: {model_type}")
    model.load_state_dict(payload["state_dict"])
    model.to(resolved_device).eval()
    return {
        "model": model,
        "model_type": model_type,
        "classes": classes,
        "input_size": int(payload["input_size"]),
        "parameter_count": int(payload["parameter_count"]),
        "framework": "pytorch",
        "device": resolved_device,
    }


def export_torch_model_to_onnx(
    result: dict[str, object],
    output_path: Path,
) -> Path:
    """Export a trained wafer model with a dynamic batch axis."""
    if not onnx_is_available():
        raise ImportError("ONNX is required for export. Install requirements-ai.txt.")
    torch, _ = _require_torch()
    model = result["model"]
    input_size = int(result["input_size"])
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    original_device = next(model.parameters()).device
    model.to("cpu").eval()
    dummy_input = torch.zeros(1, 1, input_size, input_size, dtype=torch.float32)
    try:
        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            input_names=["wafer_map"],
            output_names=["output"],
            dynamic_axes={"wafer_map": {0: "batch"}, "output": {0: "batch"}},
            opset_version=17,
            dynamo=False,
        )
    finally:
        model.to(original_device)
    return output_path
