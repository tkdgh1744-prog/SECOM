"""Wafer map spatial analytics and optional AI workflows."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
import struct
from typing import Iterable
import zlib

import numpy as np
import pandas as pd


KNOWN_WAFER_PATTERNS = [
    "Center",
    "Edge-Ring",
    "Edge-Local",
    "Scratch",
    "Donut",
    "Local",
    "Random",
    "Near-Full",
]


@dataclass(frozen=True)
class WaferMapRecord:
    """A single wafer map and optional pattern label."""

    wafer_id: str
    wafer_map: np.ndarray
    label: str | None = None


def normalize_pattern_label(value: object) -> str | None:
    """Normalize WM-811K style nested labels into a plain string."""
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
        flat = np.asarray(value, dtype=object).ravel()
        if len(flat) == 0:
            return None
        value = flat[0]
    text = str(value).strip()
    if text in {"", "[]", "none", "None", "nan"}:
        return None
    return text.strip("[]'\"")


def infer_wafer_masks(
    wafer_map: np.ndarray,
    defect_value: int | float = 2,
    valid_values: Iterable[int | float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Infer valid-die and defective-die masks from a wafer map array.

    WM-811K commonly uses 0=outside wafer, 1=normal die, 2=failed die. Binary
    maps are interpreted as 1=failed die and every finite cell as valid.
    """
    array = np.asarray(wafer_map)
    if array.ndim != 2:
        raise ValueError("Each wafer map must be a 2D array.")

    finite = np.isfinite(array.astype(float, copy=False))
    if valid_values is not None:
        valid_mask = np.isin(array, list(valid_values)) & finite
    else:
        unique_values = set(np.unique(array[finite]).tolist())
        if defect_value in unique_values:
            valid_mask = (array > 0) & finite
        elif unique_values.issubset({0, 1, 0.0, 1.0}):
            valid_mask = finite
        else:
            valid_mask = finite

    if defect_value in set(np.unique(array[finite]).tolist()):
        defect_mask = (array == defect_value) & valid_mask
    elif set(np.unique(array[finite]).tolist()).issubset({0, 1, 0.0, 1.0}):
        defect_mask = (array == 1) & valid_mask
    else:
        defect_mask = (array == defect_value) & valid_mask
    return valid_mask.astype(bool), defect_mask.astype(bool)


def connected_component_sizes(mask: np.ndarray) -> list[int]:
    """Return 8-neighbor connected component sizes for a boolean defect mask."""
    defect_mask = np.asarray(mask, dtype=bool)
    visited = np.zeros(defect_mask.shape, dtype=bool)
    sizes: list[int] = []
    rows, cols = defect_mask.shape
    neighbors = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ]

    for row in range(rows):
        for col in range(cols):
            if visited[row, col] or not defect_mask[row, col]:
                continue
            visited[row, col] = True
            queue: deque[tuple[int, int]] = deque([(row, col)])
            size = 0
            while queue:
                current_row, current_col = queue.popleft()
                size += 1
                for row_delta, col_delta in neighbors:
                    next_row = current_row + row_delta
                    next_col = current_col + col_delta
                    if (
                        0 <= next_row < rows
                        and 0 <= next_col < cols
                        and not visited[next_row, next_col]
                        and defect_mask[next_row, next_col]
                    ):
                        visited[next_row, next_col] = True
                        queue.append((next_row, next_col))
            sizes.append(size)
    return sizes


def _safe_ratio(numerator: float, denominator: float) -> float:
    return 0.0 if denominator == 0 else float(numerator) / float(denominator)


def _write_rgb_png(output_path: Path, image: np.ndarray) -> Path:
    """Write an RGB PNG using only the Python standard library."""
    rgb = np.asarray(image, dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("PNG fallback expects an RGB image array.")
    height, width, _ = rgb.shape

    def chunk(chunk_type: bytes, data: bytes) -> bytes:
        checksum = zlib.crc32(chunk_type + data) & 0xFFFFFFFF
        return struct.pack("!I", len(data)) + chunk_type + data + struct.pack("!I", checksum)

    raw = b"".join(b"\x00" + rgb[row].tobytes() for row in range(height))
    png = b"\x89PNG\r\n\x1a\n"
    png += chunk("IHDR".encode("ascii"), struct.pack("!IIBBBBB", width, height, 8, 2, 0, 0, 0))
    png += chunk("IDAT".encode("ascii"), zlib.compress(raw))
    png += chunk("IEND".encode("ascii"), b"")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(png)
    return output_path

def _jaccard_similarity(mask: np.ndarray, other: np.ndarray) -> float:
    union = np.logical_or(mask, other).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(mask, other).sum() / union)


def _radial_coordinates(valid_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows, cols = np.indices(valid_mask.shape)
    valid_rows = rows[valid_mask]
    valid_cols = cols[valid_mask]
    if len(valid_rows) == 0:
        center_row = (valid_mask.shape[0] - 1) / 2.0
        center_col = (valid_mask.shape[1] - 1) / 2.0
    else:
        center_row = (valid_rows.min() + valid_rows.max()) / 2.0
        center_col = (valid_cols.min() + valid_cols.max()) / 2.0

    row_centered = rows - center_row
    col_centered = cols - center_col
    radius = np.sqrt(row_centered**2 + col_centered**2)
    max_radius = radius[valid_mask].max() if valid_mask.any() else radius.max()
    radius_norm = np.zeros_like(radius, dtype=float) if max_radius == 0 else radius / max_radius
    angles = np.arctan2(row_centered, col_centered)
    return radius_norm, angles, col_centered


def _elongation(defect_mask: np.ndarray) -> float:
    row_idx, col_idx = np.where(defect_mask)
    if len(row_idx) < 3:
        return 0.0
    points = np.column_stack([row_idx, col_idx]).astype(float)
    covariance = np.cov(points, rowvar=False)
    eigenvalues = np.linalg.eigvalsh(covariance)
    smallest = max(float(eigenvalues[0]), 1e-9)
    return float(eigenvalues[-1] / smallest)


def extract_wafer_map_features(
    record: WaferMapRecord,
    defect_value: int | float = 2,
    valid_values: Iterable[int | float] | None = None,
    radial_bins: int = 5,
) -> dict[str, object]:
    """Extract wafer-level spatial features from one wafer map."""
    valid_mask, defect_mask = infer_wafer_masks(record.wafer_map, defect_value=defect_value, valid_values=valid_values)
    valid_die_count = int(valid_mask.sum())
    defect_die_count = int(defect_mask.sum())
    radius_norm, angles, col_centered = _radial_coordinates(valid_mask)
    defect_radii = radius_norm[defect_mask]
    defect_angles = angles[defect_mask]

    zone_masks = {
        "center": radius_norm <= 0.33,
        "middle": (radius_norm > 0.33) & (radius_norm < 0.75),
        "edge": radius_norm >= 0.75,
    }
    features: dict[str, object] = {
        "wafer_id": record.wafer_id,
        "source_label": record.label,
        "map_rows": int(np.asarray(record.wafer_map).shape[0]),
        "map_cols": int(np.asarray(record.wafer_map).shape[1]),
        "valid_die_count": valid_die_count,
        "defect_die_count": defect_die_count,
        "defect_ratio": _safe_ratio(defect_die_count, valid_die_count),
        "radius_defect_mean": float(defect_radii.mean()) if defect_die_count else 0.0,
        "radius_defect_std": float(defect_radii.std()) if defect_die_count else 0.0,
    }

    for zone, zone_mask in zone_masks.items():
        valid_in_zone = int((zone_mask & valid_mask).sum())
        defects_in_zone = int((zone_mask & defect_mask).sum())
        features[f"{zone}_defect_count"] = defects_in_zone
        features[f"{zone}_defect_ratio"] = _safe_ratio(defects_in_zone, defect_die_count)
        features[f"{zone}_defect_density"] = _safe_ratio(defects_in_zone, valid_in_zone)

    for bin_index in range(radial_bins):
        lower = bin_index / radial_bins
        upper = (bin_index + 1) / radial_bins
        if bin_index == radial_bins - 1:
            bin_mask = (radius_norm >= lower) & (radius_norm <= upper)
        else:
            bin_mask = (radius_norm >= lower) & (radius_norm < upper)
        valid_in_bin = int((bin_mask & valid_mask).sum())
        defects_in_bin = int((bin_mask & defect_mask).sum())
        features[f"radial_bin_{bin_index + 1}_defect_ratio"] = _safe_ratio(defects_in_bin, defect_die_count)
        features[f"radial_bin_{bin_index + 1}_defect_density"] = _safe_ratio(defects_in_bin, valid_in_bin)

    component_sizes = connected_component_sizes(defect_mask)
    max_cluster_size = max(component_sizes) if component_sizes else 0
    features.update(
        {
            "cluster_count": len(component_sizes),
            "max_cluster_size": int(max_cluster_size),
            "max_cluster_ratio": _safe_ratio(max_cluster_size, defect_die_count),
            "mean_cluster_size": float(np.mean(component_sizes)) if component_sizes else 0.0,
            "horizontal_symmetry": _jaccard_similarity(defect_mask, np.fliplr(defect_mask)),
            "vertical_symmetry": _jaccard_similarity(defect_mask, np.flipud(defect_mask)),
            "rotational_symmetry": _jaccard_similarity(defect_mask, np.rot90(defect_mask, 2)),
            "elongation": _elongation(defect_mask),
        }
    )

    if defect_die_count:
        vector_x = float(np.cos(defect_angles).mean())
        vector_y = float(np.sin(defect_angles).mean())
        quadrant_counts = [
            int(((defect_mask) & (col_centered >= 0) & (angles >= 0)).sum()),
            int(((defect_mask) & (col_centered < 0) & (angles >= 0)).sum()),
            int(((defect_mask) & (col_centered < 0) & (angles < 0)).sum()),
            int(((defect_mask) & (col_centered >= 0) & (angles < 0)).sum()),
        ]
    else:
        vector_x = vector_y = 0.0
        quadrant_counts = [0, 0, 0, 0]
    quadrant_ratios = [count / defect_die_count if defect_die_count else 0.0 for count in quadrant_counts]
    for index, ratio in enumerate(quadrant_ratios, start=1):
        features[f"quadrant_{index}_defect_ratio"] = ratio
    features["quadrant_imbalance"] = max(quadrant_ratios) - min(quadrant_ratios) if quadrant_ratios else 0.0
    features["direction_bias"] = float(np.sqrt(vector_x**2 + vector_y**2))
    features["direction_angle_degrees"] = float(np.degrees(np.arctan2(vector_y, vector_x))) if defect_die_count else 0.0
    features["heuristic_pattern"] = classify_wafer_pattern(features)
    return features


def classify_wafer_pattern(features: dict[str, object] | pd.Series) -> str:
    """Classify a wafer map into a common spatial defect pattern."""
    row = dict(features)
    defect_count = float(row.get("defect_die_count", 0.0))
    defect_ratio = float(row.get("defect_ratio", 0.0))
    center = float(row.get("center_defect_ratio", 0.0))
    middle = float(row.get("middle_defect_ratio", 0.0))
    edge = float(row.get("edge_defect_ratio", 0.0))
    bias = float(row.get("direction_bias", 0.0))
    imbalance = float(row.get("quadrant_imbalance", 0.0))
    max_cluster_ratio = float(row.get("max_cluster_ratio", 0.0))
    elongation = float(row.get("elongation", 0.0))
    cluster_count = float(row.get("cluster_count", 0.0))

    if defect_count == 0:
        return "Random"
    if defect_ratio >= 0.55:
        return "Near-Full"
    if defect_count >= 3 and elongation >= 6.0 and max_cluster_ratio >= 0.30:
        return "Scratch"
    if center >= 0.50 and float(row.get("radius_defect_mean", 1.0)) <= 0.45:
        return "Center"
    if middle >= 0.55 and center <= 0.20 and edge <= 0.30:
        return "Donut"
    if edge >= 0.55 and bias < 0.35 and imbalance < 0.45:
        return "Edge-Ring"
    if edge >= 0.45 and (bias >= 0.35 or imbalance >= 0.45):
        return "Edge-Local"
    if defect_count >= 3 and (max_cluster_ratio >= 0.50 or cluster_count <= 2):
        return "Local"
    return "Random"


def extract_wafer_map_feature_table(
    records: list[WaferMapRecord],
    defect_value: int | float = 2,
    valid_values: Iterable[int | float] | None = None,
    radial_bins: int = 5,
) -> pd.DataFrame:
    """Extract spatial features for all wafer maps."""
    rows = [
        extract_wafer_map_features(
            record,
            defect_value=defect_value,
            valid_values=valid_values,
            radial_bins=radial_bins,
        )
        for record in records
    ]
    return pd.DataFrame(rows)


def load_wm811k_dataframe(
    path: Path,
    wafer_map_col: str = "waferMap",
    label_col: str | None = "failureType",
    id_col: str | None = None,
) -> list[WaferMapRecord]:
    """Load WM-811K style pickle data with a waferMap column."""
    data = pd.read_pickle(path)
    if wafer_map_col not in data.columns:
        raise ValueError(f"Missing wafer map column: {wafer_map_col}")

    records: list[WaferMapRecord] = []
    for index, row in data.iterrows():
        wafer_id = str(row[id_col]) if id_col and id_col in data.columns else str(index)
        label = normalize_pattern_label(row[label_col]) if label_col and label_col in data.columns else None
        records.append(WaferMapRecord(wafer_id=wafer_id, wafer_map=np.asarray(row[wafer_map_col]), label=label))
    return records


def load_array_wafer_maps(path: Path) -> list[WaferMapRecord]:
    """Load 2D/3D wafer maps from .npy or .npz files."""
    if path.suffix.lower() == ".npz":
        archive = np.load(path, allow_pickle=True)
        key = "wafer_maps" if "wafer_maps" in archive else archive.files[0]
        maps = archive[key]
        ids = archive["wafer_ids"] if "wafer_ids" in archive else np.arange(len(maps))
        labels = archive["labels"] if "labels" in archive else [None] * len(maps)
    else:
        maps = np.load(path, allow_pickle=True)
        ids = np.arange(len(maps)) if np.asarray(maps).ndim >= 3 else [0]
        labels = [None] * len(ids)
        if np.asarray(maps).ndim == 2:
            maps = np.asarray([maps])

    return [
        WaferMapRecord(wafer_id=str(wafer_id), wafer_map=np.asarray(wafer_map), label=normalize_pattern_label(label))
        for wafer_id, wafer_map, label in zip(ids, maps, labels)
    ]


def coordinate_table_to_wafer_records(
    df: pd.DataFrame,
    wafer_id_col: str = "wafer_id",
    x_col: str = "x",
    y_col: str = "y",
    value_col: str | None = None,
    defect_col: str | None = None,
    label_col: str | None = None,
    defect_value: int | float | str = 1,
) -> list[WaferMapRecord]:
    """Convert coordinate-level die inspection rows into wafer map records."""
    required = {wafer_id_col, x_col, y_col}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Missing coordinate wafer columns: {missing}")

    records: list[WaferMapRecord] = []
    for wafer_id, group in df.groupby(wafer_id_col, dropna=False):
        x_values = group[x_col].astype(int)
        y_values = group[y_col].astype(int)
        x_min, y_min = int(x_values.min()), int(y_values.min())
        x_shifted = (x_values - x_min).to_numpy()
        y_shifted = (y_values - y_min).to_numpy()
        wafer_map = np.zeros((int(y_shifted.max()) + 1, int(x_shifted.max()) + 1), dtype=int)

        if value_col and value_col in group.columns:
            values = group[value_col].to_numpy()
            wafer_map[y_shifted, x_shifted] = np.where(values == defect_value, 2, 1)
        elif defect_col and defect_col in group.columns:
            values = group[defect_col].astype(bool).to_numpy()
            wafer_map[y_shifted, x_shifted] = np.where(values, 2, 1)
        else:
            wafer_map[y_shifted, x_shifted] = 2

        label = normalize_pattern_label(group[label_col].iloc[0]) if label_col and label_col in group.columns else None
        records.append(WaferMapRecord(wafer_id=str(wafer_id), wafer_map=wafer_map, label=label))
    return records


def load_wafer_map_records(
    path: Path,
    input_format: str = "auto",
    wafer_map_col: str = "waferMap",
    label_col: str | None = "failureType",
    id_col: str | None = None,
    wafer_id_col: str = "wafer_id",
    x_col: str = "x",
    y_col: str = "y",
    value_col: str | None = None,
    defect_col: str | None = None,
    coordinate_defect_value: int | float | str = 1,
) -> list[WaferMapRecord]:
    """Load wafer map records from WM-811K pickle, array files, or coordinate CSV."""
    path = Path(path)
    detected_format = input_format
    if detected_format == "auto":
        if path.suffix.lower() in {".pkl", ".pickle"}:
            detected_format = "wm811k"
        elif path.suffix.lower() in {".npy", ".npz"}:
            detected_format = "array"
        elif path.suffix.lower() == ".csv":
            detected_format = "coordinate"
        else:
            raise ValueError(f"Cannot infer wafer map input format from: {path}")

    if detected_format == "wm811k":
        return load_wm811k_dataframe(path, wafer_map_col=wafer_map_col, label_col=label_col, id_col=id_col)
    if detected_format == "array":
        return load_array_wafer_maps(path)
    if detected_format == "coordinate":
        df = pd.read_csv(path)
        return coordinate_table_to_wafer_records(
            df,
            wafer_id_col=wafer_id_col,
            x_col=x_col,
            y_col=y_col,
            value_col=value_col,
            defect_col=defect_col,
            label_col=label_col,
            defect_value=coordinate_defect_value,
        )
    raise ValueError(f"Unsupported wafer map input format: {input_format}")


def resize_binary_map(mask: np.ndarray, size: int = 32) -> np.ndarray:
    """Resize a boolean wafer map with nearest-neighbor sampling."""
    if size <= 0:
        raise ValueError("size must be positive.")
    source = np.asarray(mask, dtype=float)
    row_idx = np.linspace(0, source.shape[0] - 1, size).round().astype(int)
    col_idx = np.linspace(0, source.shape[1] - 1, size).round().astype(int)
    return source[np.ix_(row_idx, col_idx)]


def wafer_similarity_pairs(
    records: list[WaferMapRecord],
    top_k: int = 3,
    defect_value: int | float = 2,
    resize_to: int = 32,
) -> pd.DataFrame:
    """Compute top-k cosine-similar wafer map pairs."""
    if len(records) < 2:
        return pd.DataFrame(columns=["wafer_id", "similar_wafer_id", "similarity", "rank"])

    vectors = []
    for record in records:
        _, defect_mask = infer_wafer_masks(record.wafer_map, defect_value=defect_value)
        vectors.append(resize_binary_map(defect_mask, size=resize_to).ravel())
    matrix = np.vstack(vectors)
    norms = np.linalg.norm(matrix, axis=1)
    denominator = np.outer(norms, norms)
    similarity = np.divide(matrix @ matrix.T, denominator, out=np.zeros((len(records), len(records))), where=denominator != 0)
    np.fill_diagonal(similarity, -1.0)

    rows = []
    for index, record in enumerate(records):
        neighbor_indices = np.argsort(similarity[index])[::-1][:top_k]
        for rank, neighbor_index in enumerate(neighbor_indices, start=1):
            rows.append(
                {
                    "wafer_id": record.wafer_id,
                    "similar_wafer_id": records[int(neighbor_index)].wafer_id,
                    "similarity": float(similarity[index, neighbor_index]),
                    "rank": rank,
                }
            )
    return pd.DataFrame(rows)


def cluster_wafer_patterns(features: pd.DataFrame, n_clusters: int = 8, random_state: int = 42) -> pd.Series:
    """Cluster unknown wafer patterns from engineered numeric features."""
    if features.empty:
        return pd.Series(dtype=int, name="pattern_cluster")

    try:
        from sklearn.cluster import KMeans
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        pattern_codes = pd.Categorical(features.get("heuristic_pattern", pd.Series(index=features.index))).codes
        fallback_labels = np.mod(np.maximum(pattern_codes, 0), max(1, n_clusters))
        return pd.Series(fallback_labels, index=features.index, name="pattern_cluster")

    numeric = features.select_dtypes(include=[np.number]).fillna(0.0)
    usable_clusters = min(max(1, n_clusters), len(numeric))
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("cluster", KMeans(n_clusters=usable_clusters, n_init=10, random_state=random_state)),
        ]
    )
    labels = model.fit_predict(numeric)
    return pd.Series(labels, index=features.index, name="pattern_cluster")


def write_wafer_map_image(
    record: WaferMapRecord,
    output_path: Path,
    defect_value: int | float = 2,
    title: str | None = None,
) -> Path:
    """Write a PNG visualization for one wafer map."""
    valid_mask, defect_mask = infer_wafer_masks(record.wafer_map, defect_value=defect_value)
    display = np.zeros(valid_mask.shape, dtype=int)
    display[valid_mask] = 1
    display[defect_mask] = 2
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
    except ImportError:
        palette = np.array(
            [
                [17, 24, 39],
                [209, 213, 219],
                [239, 68, 68],
            ],
            dtype=np.uint8,
        )
        rgb = palette[display]
        scale = max(2, min(12, 240 // max(1, max(rgb.shape[0], rgb.shape[1]))))
        return _write_rgb_png(output_path, np.repeat(np.repeat(rgb, scale, axis=0), scale, axis=1))

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(display, interpolation="nearest", cmap=ListedColormap(["#111827", "#d1d5db", "#ef4444"]))
    ax.set_title(title or f"Wafer {record.wafer_id}")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path

def write_pattern_summary_chart(summary: pd.DataFrame, output_path: Path) -> Path:
    """Write a bar chart of detected wafer map patterns."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        width, height = 640, 320
        image = np.full((height, width, 3), 255, dtype=np.uint8)
        if not summary.empty:
            counts = summary["wafer_count"].to_numpy(dtype=float)
            max_count = max(float(counts.max()), 1.0)
            bar_count = len(counts)
            gap = 8
            bar_width = max(8, (width - gap * (bar_count + 1)) // max(1, bar_count))
            for index, count in enumerate(counts):
                x0 = gap + index * (bar_width + gap)
                x1 = min(width - gap, x0 + bar_width)
                bar_height = int((height - 60) * count / max_count)
                y0 = height - 30 - bar_height
                image[y0 : height - 30, x0:x1] = np.array([37, 99, 235], dtype=np.uint8)
            image[height - 30 : height - 27, 20 : width - 20] = np.array([31, 41, 55], dtype=np.uint8)
        return _write_rgb_png(output_path, image)

    fig, ax = plt.subplots(figsize=(8, 4))
    if not summary.empty:
        ax.bar(summary["heuristic_pattern"], summary["wafer_count"], color="#2563eb")
    ax.set_xlabel("Pattern")
    ax.set_ylabel("Wafer count")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path

def _dataframe_to_markdown(df: pd.DataFrame, max_rows: int = 20) -> str:
    """Render a small markdown table without optional tabulate dependency."""
    if df.empty:
        return "_No rows._"
    display_df = df.head(max_rows).copy()
    headers = [str(column) for column in display_df.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in display_df.to_numpy():
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)

def build_wafer_map_report(
    features: pd.DataFrame,
    pattern_summary: pd.DataFrame,
    similarity_pairs: pd.DataFrame,
    output_path: Path,
) -> str:
    """Build a Markdown wafer map analysis report."""
    sections = [
        "# Wafer Map Spatial Defect Analysis",
        (
            "This report summarizes spatial defect signatures from wafer maps: die-level defect burden, "
            "center/edge concentration, radial distribution, connected defect clusters, symmetry, "
            "directional bias, heuristic pattern classes, similar wafer pairs, and unsupervised clusters."
        ),
        "## Pattern Summary",
        _dataframe_to_markdown(pattern_summary),
        "## Highest Defect Ratio Wafers",
        _dataframe_to_markdown(
            features.sort_values("defect_ratio", ascending=False)[
                ["wafer_id", "defect_ratio", "defect_die_count", "heuristic_pattern", "source_label"]
            ]
        )
        if not features.empty
        else "_No wafer rows._",
        "## Similar Wafer Pairs",
        _dataframe_to_markdown(similarity_pairs),
    ]
    report = "\n\n".join(sections) + "\n"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    return report


def train_cnn_pattern_classifier(
    records: list[WaferMapRecord],
    labels: list[str],
    epochs: int = 5,
    resize_to: int = 32,
    random_state: int = 42,
) -> dict[str, object]:
    """Train a small CNN pattern classifier when TensorFlow is available."""
    try:
        import tensorflow as tf
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder
    except ImportError as exc:
        raise ImportError("TensorFlow and scikit-learn are required for CNN training.") from exc

    tf.keras.utils.set_random_seed(random_state)
    label_encoder = LabelEncoder()
    encoded = label_encoder.fit_transform(labels)
    images = []
    for record in records:
        _, defect_mask = infer_wafer_masks(record.wafer_map)
        images.append(resize_binary_map(defect_mask, size=resize_to))
    X = np.asarray(images, dtype="float32")[..., np.newaxis]
    y = tf.keras.utils.to_categorical(encoded)

    stratify = encoded if len(np.unique(encoded)) > 1 and min(np.bincount(encoded)) >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=random_state,
        stratify=stratify,
    )
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(resize_to, resize_to, 1)),
            tf.keras.layers.Conv2D(16, 3, activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(len(label_encoder.classes_), activation="softmax"),
        ]
    )
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    history = model.fit(X_train, y_train, validation_split=0.2, epochs=epochs, verbose=0)
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    return {
        "model": model,
        "label_encoder": label_encoder,
        "history": history.history,
        "test_loss": float(loss),
        "test_accuracy": float(accuracy),
        "classes": label_encoder.classes_.tolist(),
    }


def train_autoencoder_anomaly_scores(
    records: list[WaferMapRecord],
    epochs: int = 5,
    resize_to: int = 32,
    random_state: int = 42,
) -> pd.DataFrame:
    """Train a compact convolutional autoencoder and return reconstruction errors."""
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise ImportError("TensorFlow is required for autoencoder anomaly detection.") from exc

    tf.keras.utils.set_random_seed(random_state)
    images = []
    for record in records:
        _, defect_mask = infer_wafer_masks(record.wafer_map)
        images.append(resize_binary_map(defect_mask, size=resize_to))
    X = np.asarray(images, dtype="float32")[..., np.newaxis]

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(resize_to, resize_to, 1)),
            tf.keras.layers.Conv2D(16, 3, activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D(padding="same"),
            tf.keras.layers.Conv2D(8, 3, activation="relu", padding="same"),
            tf.keras.layers.UpSampling2D(),
            tf.keras.layers.Conv2D(16, 3, activation="relu", padding="same"),
            tf.keras.layers.UpSampling2D(),
            tf.keras.layers.Conv2D(1, 3, activation="sigmoid", padding="same"),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    model.fit(X, X, epochs=epochs, validation_split=0.1 if len(X) >= 10 else 0.0, verbose=0)
    reconstruction = model.predict(X, verbose=0)
    errors = ((X - reconstruction) ** 2).mean(axis=(1, 2, 3))
    return pd.DataFrame(
        {
            "wafer_id": [record.wafer_id for record in records],
            "autoencoder_reconstruction_error": errors.astype(float),
        }
    ).sort_values("autoencoder_reconstruction_error", ascending=False)


def run_wafer_map_analysis(
    records: list[WaferMapRecord],
    output_dir: Path = Path("outputs/wafer_maps"),
    defect_value: int | float = 2,
    n_clusters: int = 8,
    top_k: int = 3,
    max_images: int = 12,
) -> dict[str, Path]:
    """Run the wafer map analysis workflow and write artifacts."""
    if not records:
        raise ValueError("At least one wafer map record is required.")

    output_dir = Path(output_dir)
    image_dir = output_dir / "images"
    features = extract_wafer_map_feature_table(records, defect_value=defect_value)
    features["pattern_cluster"] = cluster_wafer_patterns(features, n_clusters=n_clusters)
    pattern_summary = (
        features.groupby("heuristic_pattern", dropna=False)
        .agg(
            wafer_count=("wafer_id", "size"),
            mean_defect_ratio=("defect_ratio", "mean"),
            mean_cluster_count=("cluster_count", "mean"),
            mean_direction_bias=("direction_bias", "mean"),
        )
        .reset_index()
        .sort_values(["wafer_count", "heuristic_pattern"], ascending=[False, True])
    )
    cluster_summary = (
        features.groupby("pattern_cluster", dropna=False)
        .agg(
            wafer_count=("wafer_id", "size"),
            dominant_heuristic_pattern=("heuristic_pattern", lambda values: values.mode().iloc[0]),
            mean_defect_ratio=("defect_ratio", "mean"),
            mean_cluster_count=("cluster_count", "mean"),
        )
        .reset_index()
    )
    similarity_pairs = wafer_similarity_pairs(records, top_k=top_k, defect_value=defect_value)

    output_dir.mkdir(parents=True, exist_ok=True)
    feature_path = output_dir / "wafer_map_features.csv"
    pattern_summary_path = output_dir / "pattern_summary.csv"
    cluster_summary_path = output_dir / "cluster_summary.csv"
    similarity_path = output_dir / "similar_wafer_pairs.csv"
    report_path = output_dir / "wafer_map_report.md"
    chart_path = image_dir / "pattern_summary.png"

    features.to_csv(feature_path, index=False)
    pattern_summary.to_csv(pattern_summary_path, index=False)
    cluster_summary.to_csv(cluster_summary_path, index=False)
    similarity_pairs.to_csv(similarity_path, index=False)
    write_pattern_summary_chart(pattern_summary, chart_path)
    for record in sorted(records, key=lambda item: str(item.wafer_id))[:max_images]:
        write_wafer_map_image(record, image_dir / f"wafer_{record.wafer_id}.png", defect_value=defect_value)
    build_wafer_map_report(features, pattern_summary, similarity_pairs, report_path)

    return {
        "features": feature_path,
        "pattern_summary": pattern_summary_path,
        "cluster_summary": cluster_summary_path,
        "similarity_pairs": similarity_path,
        "report": report_path,
        "pattern_chart": chart_path,
    }









