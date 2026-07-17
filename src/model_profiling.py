"""Repeatable CPU profiling for PyTorch and ONNX wafer models."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any, Callable

import numpy as np
import pandas as pd

from src.wafer_torch import load_torch_model_bundle


def summarize_latencies(latencies_ms: list[float]) -> dict[str, float]:
    """Summarize repeated latency observations in milliseconds."""
    if not latencies_ms:
        raise ValueError("At least one latency observation is required.")
    values = np.asarray(latencies_ms, dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Latency observations must be finite and non-negative.")
    return {
        "latency_mean_ms": float(values.mean()),
        "latency_p50_ms": float(np.percentile(values, 50)),
        "latency_p95_ms": float(np.percentile(values, 95)),
        "latency_min_ms": float(values.min()),
        "latency_max_ms": float(values.max()),
        "latency_std_ms": float(values.std()),
    }


def _validate_measurement_settings(
    batch_size: int,
    warmup_runs: int,
    measured_runs: int,
) -> None:
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0.")
    if warmup_runs < 0:
        raise ValueError("warmup_runs must be 0 or greater.")
    if measured_runs <= 0:
        raise ValueError("measured_runs must be greater than 0.")


def _package_version(distribution: str) -> str | None:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def runtime_environment() -> dict[str, object]:
    """Return machine and runtime metadata needed to interpret benchmarks."""
    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "python_version": platform.python_version(),
        "pytorch_version": _package_version("torch"),
        "onnx_version": _package_version("onnx"),
        "onnxruntime_version": _package_version("onnxruntime"),
    }


def _process_rss_bytes() -> int | None:
    try:
        import psutil

        return int(psutil.Process().memory_info().rss)
    except (ImportError, OSError):
        pass

    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes

            class ProcessMemoryCounters(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            counters = ProcessMemoryCounters()
            counters.cb = ctypes.sizeof(counters)
            get_current_process = ctypes.windll.kernel32.GetCurrentProcess
            get_current_process.restype = wintypes.HANDLE
            get_process_memory_info = ctypes.windll.psapi.GetProcessMemoryInfo
            get_process_memory_info.argtypes = [
                wintypes.HANDLE,
                ctypes.POINTER(ProcessMemoryCounters),
                wintypes.DWORD,
            ]
            get_process_memory_info.restype = wintypes.BOOL
            process = get_current_process()
            success = get_process_memory_info(
                process,
                ctypes.byref(counters),
                counters.cb,
            )
            return int(counters.WorkingSetSize) if success else None
        except (AttributeError, OSError):
            return None

    statm_path = Path("/proc/self/statm")
    if statm_path.exists():
        try:
            resident_pages = int(statm_path.read_text(encoding="ascii").split()[1])
            return resident_pages * int(os.sysconf("SC_PAGE_SIZE"))
        except (OSError, ValueError, IndexError, AttributeError):
            return None
    return None


def _measure_inference(
    run_once: Callable[[], Any],
    warmup_runs: int,
    measured_runs: int,
    synchronize: Callable[[], None] | None = None,
) -> tuple[dict[str, float], Any, dict[str, int | None]]:
    sync = synchronize or (lambda: None)
    output: Any = None
    for _ in range(warmup_runs):
        output = run_once()
        sync()

    rss_before = _process_rss_bytes()
    rss_observations = [rss_before] if rss_before is not None else []
    latencies_ms: list[float] = []
    for _ in range(measured_runs):
        sync()
        started_ns = time.perf_counter_ns()
        output = run_once()
        sync()
        latencies_ms.append((time.perf_counter_ns() - started_ns) / 1_000_000)
        rss = _process_rss_bytes()
        if rss is not None:
            rss_observations.append(rss)
    rss_after = _process_rss_bytes()
    if rss_after is not None:
        rss_observations.append(rss_after)
    memory = {
        "process_rss_before_bytes": rss_before,
        "process_rss_after_bytes": rss_after,
        "process_rss_max_observed_bytes": max(rss_observations) if rss_observations else None,
        "process_rss_delta_bytes": (
            rss_after - rss_before
            if rss_before is not None and rss_after is not None
            else None
        ),
    }
    return summarize_latencies(latencies_ms), output, memory


def profile_pytorch_bundle(
    model_path: Path,
    batch_size: int = 1,
    warmup_runs: int = 10,
    measured_runs: int = 50,
    device: str = "cpu",
) -> dict[str, object]:
    """Profile a saved wafer PyTorch bundle with synthetic input."""
    _validate_measurement_settings(batch_size, warmup_runs, measured_runs)
    import torch

    model_path = Path(model_path)
    bundle = load_torch_model_bundle(model_path, device=device)
    model = bundle["model"]
    resolved_device = str(bundle["device"])
    input_size = int(bundle["input_size"])
    inputs = torch.zeros(
        batch_size,
        1,
        input_size,
        input_size,
        dtype=torch.float32,
        device=resolved_device,
    )

    def run_once() -> Any:
        with torch.inference_mode():
            return model(inputs)

    def synchronize() -> None:
        if resolved_device == "cuda":
            torch.cuda.synchronize()

    latency, output, memory = _measure_inference(
        run_once,
        warmup_runs=warmup_runs,
        measured_runs=measured_runs,
        synchronize=synchronize,
    )
    parameters = list(model.parameters())
    parameter_bytes = sum(parameter.numel() * parameter.element_size() for parameter in parameters)
    operator_counts = Counter(
        type(module).__name__
        for module in model.modules()
        if module is not model and not list(module.children())
    )
    mean_latency = latency["latency_mean_ms"]
    return {
        "model_name": model_path.stem,
        "model_type": bundle["model_type"],
        "runtime": "pytorch",
        "runtime_version": str(torch.__version__),
        "precision": "FP32",
        "device": resolved_device,
        "provider": resolved_device,
        "model_path": str(model_path.resolve()),
        "model_size_bytes": model_path.stat().st_size,
        "parameter_count": int(bundle["parameter_count"]),
        "parameter_bytes": int(parameter_bytes),
        "operation_node_count": int(sum(operator_counts.values())),
        "operator_counts": dict(sorted(operator_counts.items())),
        "input_shape": list(inputs.shape),
        "input_bytes": int(inputs.numel() * inputs.element_size()),
        "output_bytes": int(output.numel() * output.element_size()),
        "batch_size": batch_size,
        "warmup_runs": warmup_runs,
        "measured_runs": measured_runs,
        "throughput_items_per_second": float(batch_size * 1000 / mean_latency),
        **latency,
        **memory,
    }


def _onnx_tensor_shape(
    declared_shape: list[Any],
    batch_size: int,
    input_size: int | None,
) -> list[int]:
    resolved: list[int] = []
    for index, dimension in enumerate(declared_shape):
        if index == 0:
            resolved.append(batch_size)
        elif isinstance(dimension, int) and dimension > 0:
            resolved.append(dimension)
        elif input_size is not None and index >= 2:
            resolved.append(input_size)
        else:
            raise ValueError(
                "The ONNX input has a dynamic non-batch dimension; pass input_size."
            )
    return resolved


def profile_onnx_model(
    model_path: Path,
    batch_size: int = 1,
    warmup_runs: int = 10,
    measured_runs: int = 50,
    input_size: int | None = None,
    intra_op_threads: int = 1,
    precision: str = "FP32",
) -> dict[str, object]:
    """Profile an ONNX model with the CPUExecutionProvider."""
    _validate_measurement_settings(batch_size, warmup_runs, measured_runs)
    if input_size is not None and input_size <= 0:
        raise ValueError("input_size must be greater than 0 when supplied.")
    if intra_op_threads <= 0:
        raise ValueError("intra_op_threads must be greater than 0.")

    import onnx
    from onnx import numpy_helper
    import onnxruntime as ort

    model_path = Path(model_path)
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = intra_op_threads
    session_options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(model_path),
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )
    model_input = session.get_inputs()[0]
    if model_input.type != "tensor(float)":
        raise ValueError(f"Unsupported ONNX input type: {model_input.type}")
    shape = _onnx_tensor_shape(model_input.shape, batch_size, input_size)
    inputs = np.zeros(shape, dtype=np.float32)

    def run_once() -> list[np.ndarray]:
        return session.run(None, {model_input.name: inputs})

    latency, outputs, memory = _measure_inference(
        run_once,
        warmup_runs=warmup_runs,
        measured_runs=measured_runs,
    )
    graph = onnx.load(str(model_path), load_external_data=False).graph
    operator_counts = Counter(node.op_type for node in graph.node)
    parameter_count = 0
    parameter_bytes = 0
    for initializer in graph.initializer:
        array = numpy_helper.to_array(initializer)
        parameter_count += int(array.size)
        parameter_bytes += int(array.nbytes)
    mean_latency = latency["latency_mean_ms"]
    return {
        "model_name": model_path.stem,
        "model_type": "onnx_model",
        "runtime": "onnxruntime",
        "runtime_version": str(ort.__version__),
        "precision": str(precision).upper(),
        "device": "cpu",
        "provider": "CPUExecutionProvider",
        "model_path": str(model_path.resolve()),
        "model_size_bytes": model_path.stat().st_size,
        "parameter_count": parameter_count,
        "parameter_bytes": parameter_bytes,
        "operation_node_count": int(sum(operator_counts.values())),
        "operator_counts": dict(sorted(operator_counts.items())),
        "input_shape": list(inputs.shape),
        "input_bytes": int(inputs.nbytes),
        "output_bytes": int(sum(output.nbytes for output in outputs)),
        "batch_size": batch_size,
        "warmup_runs": warmup_runs,
        "measured_runs": measured_runs,
        "intra_op_threads": intra_op_threads,
        "throughput_items_per_second": float(batch_size * 1000 / mean_latency),
        **latency,
        **memory,
    }


def quantize_onnx_dynamic_int8(model_path: Path, output_path: Path) -> Path:
    """Create a weight-quantized INT8 ONNX model for CPU comparison."""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    model_path = Path(model_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    quantize_dynamic(
        str(model_path),
        str(output_path),
        weight_type=QuantType.QInt8,
    )
    return output_path


def write_profile_outputs(
    profiles: list[dict[str, object]],
    output_dir: Path,
) -> dict[str, Path]:
    """Write flat CSVs plus a lossless JSON profiling report."""
    if not profiles:
        raise ValueError("At least one model profile is required.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    profiles_path = output_dir / "model_profiles.csv"
    operations_path = output_dir / "operation_profiles.csv"
    report_path = output_dir / "model_profiles.json"

    flat_profiles: list[dict[str, object]] = []
    operation_rows: list[dict[str, object]] = []
    for profile in profiles:
        flat_profile = dict(profile)
        operator_counts = dict(flat_profile.pop("operator_counts", {}))
        flat_profile["input_shape"] = json.dumps(flat_profile.get("input_shape", []))
        flat_profile["operator_counts_json"] = json.dumps(operator_counts, sort_keys=True)
        flat_profiles.append(flat_profile)
        for operator, count in sorted(operator_counts.items()):
            operation_rows.append(
                {
                    "model_name": profile["model_name"],
                    "runtime": profile["runtime"],
                    "precision": profile["precision"],
                    "operator": operator,
                    "node_count": count,
                }
            )

    pd.DataFrame(flat_profiles).to_csv(profiles_path, index=False)
    pd.DataFrame(
        operation_rows,
        columns=["model_name", "runtime", "precision", "operator", "node_count"],
    ).to_csv(operations_path, index=False)
    report = {
        "environment": runtime_environment(),
        "profiles": profiles,
        "memory_note": (
            "RSS values are process-level observations sampled between runs; parameter, input, "
            "and output bytes are the stable model memory proxies."
        ),
    }
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return {
        "profiles_csv": profiles_path,
        "operations_csv": operations_path,
        "report_json": report_path,
    }
