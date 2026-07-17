"""WM-811K dataset provenance and structural preflight checks."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.wafer_map_analysis import normalize_pattern_label


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of a file without loading it into memory."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0.")
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalized_counts(series: pd.Series) -> tuple[dict[str, int], int]:
    values = [normalize_pattern_label(value) for value in series]
    present = [value for value in values if value is not None]
    return dict(sorted(Counter(present).items())), len(values) - len(present)


def _map_sample_summary(maps: pd.Series, sample_records: int) -> dict[str, object]:
    sampled = maps.iloc[:sample_records]
    shape_counts: Counter[str] = Counter()
    value_counts: Counter[str] = Counter()
    valid_count = 0
    invalid_count = 0
    for value in sampled:
        array = np.asarray(value)
        if array.ndim != 2 or array.size == 0:
            invalid_count += 1
            continue
        valid_count += 1
        shape_counts[f"{array.shape[0]}x{array.shape[1]}"] += 1
        try:
            numeric = array.astype(float, copy=False)
            finite = numeric[np.isfinite(numeric)]
        except (TypeError, ValueError):
            invalid_count += 1
            valid_count -= 1
            shape_counts[f"{array.shape[0]}x{array.shape[1]}"] -= 1
            continue
        for item, count in zip(*np.unique(finite, return_counts=True)):
            label = str(int(item)) if float(item).is_integer() else str(float(item))
            value_counts[label] += int(count)
    return {
        "requested_records": sample_records,
        "inspected_records": len(sampled),
        "valid_2d_maps": valid_count,
        "invalid_maps": invalid_count,
        "shape_counts": dict(sorted((key, value) for key, value in shape_counts.items() if value)),
        "die_value_counts": dict(sorted(value_counts.items())),
    }


def inspect_wm811k_dataset(
    path: Path,
    wafer_map_col: str = "waferMap",
    label_col: str | None = "failureType",
    group_col: str | None = "lotName",
    id_col: str | None = None,
    sample_records: int = 5000,
    integration_mode: str = "real",
    source_uri: str | None = None,
    license_note: str | None = None,
    compute_sha256: bool = False,
) -> dict[str, object]:
    """Inspect a trusted WM-811K pickle before analysis or training."""
    if sample_records <= 0:
        raise ValueError("sample_records must be greater than 0.")
    if integration_mode not in {"real", "synthetic"}:
        raise ValueError("integration_mode must be real or synthetic.")
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"WM-811K input file not found: {path}")

    data = pd.read_pickle(path)
    if not isinstance(data, pd.DataFrame):
        raise ValueError("WM-811K pickle must contain a pandas DataFrame.")
    if wafer_map_col not in data.columns:
        raise ValueError(f"Missing wafer map column: {wafer_map_col}")

    label_available = bool(label_col and label_col in data.columns)
    label_counts: dict[str, int] = {}
    unlabeled_count = len(data)
    if label_available:
        label_counts, unlabeled_count = _normalized_counts(data[label_col])

    group_available = bool(group_col and group_col in data.columns)
    group_counts: dict[str, int] = {}
    missing_group_count = len(data)
    if group_available:
        group_counts, missing_group_count = _normalized_counts(data[group_col])

    id_available = bool(id_col and id_col in data.columns)
    identifiers = (
        data[id_col].map(str)
        if id_available
        else pd.Series(data.index.map(str), index=data.index)
    )
    duplicate_id_count = int(identifiers.duplicated().sum())
    checksum = sha256_file(path) if compute_sha256 else None
    normalized_source = str(source_uri).strip() if source_uri else None
    normalized_license = str(license_note).strip() if license_note else None
    warnings = [
        "Pickle files can execute code when loaded; use only a trusted, verified source."
    ]
    if not label_available:
        warnings.append("No label column was available; supervised classification cannot run.")
    elif unlabeled_count:
        warnings.append(f"{unlabeled_count} records have no normalized pattern label.")
    if label_available and len(label_counts) < 2:
        warnings.append("Fewer than two labeled classes were detected.")
    if not group_available:
        warnings.append("No lot/group column was available; lot-level leakage checks cannot run.")
    elif missing_group_count:
        warnings.append(f"{missing_group_count} records have no normalized lot/group value.")
    if group_available and len(group_counts) < 2:
        warnings.append("Fewer than two lot/groups were detected; grouped evaluation cannot run.")
    if duplicate_id_count:
        warnings.append(f"{duplicate_id_count} duplicate wafer identifiers were detected.")

    map_sample = _map_sample_summary(data[wafer_map_col], sample_records)
    if map_sample["invalid_maps"]:
        warnings.append(
            f"{map_sample['invalid_maps']} sampled records are not valid numeric 2D wafer maps."
        )

    provenance_complete = bool(normalized_source and normalized_license and checksum)
    if not provenance_complete:
        warnings.append(
            "Provenance is incomplete; source URI, license note, and SHA-256 are required before final real-data claims."
        )
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "integration_mode": integration_mode,
        "input": {
            "path": str(path),
            "file_name": path.name,
            "file_size_bytes": path.stat().st_size,
            "sha256": checksum,
        },
        "provenance": {
            "source_uri": normalized_source,
            "license_note": normalized_license,
            "complete": provenance_complete,
        },
        "schema": {
            "row_count": len(data),
            "columns": [str(column) for column in data.columns],
            "wafer_map_column": wafer_map_col,
            "label_column": label_col if label_available else None,
            "group_column": group_col if group_available else None,
            "id_column": id_col if id_available else None,
        },
        "labels": {
            "labeled_count": len(data) - unlabeled_count,
            "unlabeled_count": unlabeled_count,
            "class_count": len(label_counts),
            "counts": label_counts,
        },
        "groups": {
            "group_count": len(group_counts),
            "missing_group_count": missing_group_count,
            "duplicate_id_count": duplicate_id_count,
        },
        "map_sample": map_sample,
        "ready_for_grouped_training": bool(
            integration_mode == "real"
            and provenance_complete
            and len(label_counts) >= 2
            and len(group_counts) >= 2
            and map_sample["invalid_maps"] == 0
        ),
        "warnings": warnings,
    }


def write_wm811k_preflight(report: dict[str, Any], output_path: Path) -> Path:
    """Write a machine-readable WM-811K preflight report."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return output_path
