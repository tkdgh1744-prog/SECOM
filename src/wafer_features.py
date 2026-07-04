"""Feature engineering utilities for wafer defect inspection data."""

from __future__ import annotations

import numpy as np
import pandas as pd


REQUIRED_WAFER_COLUMNS = {"wafer_id", "x", "y"}


def validate_wafer_columns(
    wafer_df: pd.DataFrame,
    wafer_id_col: str = "wafer_id",
    x_col: str = "x",
    y_col: str = "y",
) -> None:
    """Validate that a wafer inspection table has required coordinate columns."""
    required = {wafer_id_col, x_col, y_col}
    missing = sorted(required.difference(wafer_df.columns))
    if missing:
        raise ValueError(f"Missing wafer inspection columns: {missing}")


def add_radial_position(
    wafer_df: pd.DataFrame,
    x_col: str = "x",
    y_col: str = "y",
    output_col: str = "radius_norm",
) -> pd.DataFrame:
    """Add normalized radial distance from the wafer center.

    Coordinates are normalized using the maximum observed absolute x/y distance.
    This keeps the utility usable for synthetic, pixel, or die-grid coordinates.
    """
    result = wafer_df.copy()
    x_centered = result[x_col] - result[x_col].mean()
    y_centered = result[y_col] - result[y_col].mean()
    radius = np.sqrt(x_centered**2 + y_centered**2)
    max_radius = radius.max()
    result[output_col] = 0.0 if max_radius == 0 else radius / max_radius
    return result


def classify_radial_zone(radius_norm: pd.Series, center_cutoff: float = 0.33, edge_cutoff: float = 0.75) -> pd.Series:
    """Classify normalized radius into center, middle, and edge zones."""
    return pd.Series(
        np.select(
            [radius_norm <= center_cutoff, radius_norm >= edge_cutoff],
            ["center", "edge"],
            default="middle",
        ),
        index=radius_norm.index,
        name="radial_zone",
    )


def add_spatial_bins(
    wafer_df: pd.DataFrame,
    x_col: str = "x",
    y_col: str = "y",
) -> pd.DataFrame:
    """Add radial zone and quadrant columns for wafer defect coordinates."""
    result = add_radial_position(wafer_df, x_col=x_col, y_col=y_col)
    result["radial_zone"] = classify_radial_zone(result["radius_norm"])
    result["quadrant"] = np.select(
        [
            (result[x_col] >= result[x_col].mean()) & (result[y_col] >= result[y_col].mean()),
            (result[x_col] < result[x_col].mean()) & (result[y_col] >= result[y_col].mean()),
            (result[x_col] < result[x_col].mean()) & (result[y_col] < result[y_col].mean()),
            (result[x_col] >= result[x_col].mean()) & (result[y_col] < result[y_col].mean()),
        ],
        ["Q1", "Q2", "Q3", "Q4"],
        default="unknown",
    )
    return result


def wafer_defect_features(
    wafer_df: pd.DataFrame,
    wafer_id_col: str = "wafer_id",
    x_col: str = "x",
    y_col: str = "y",
    defect_type_col: str | None = "defect_type",
) -> pd.DataFrame:
    """Aggregate coordinate-level wafer defects into wafer-level features."""
    validate_wafer_columns(wafer_df, wafer_id_col=wafer_id_col, x_col=x_col, y_col=y_col)
    enriched = add_spatial_bins(wafer_df, x_col=x_col, y_col=y_col)

    grouped = enriched.groupby(wafer_id_col, dropna=False)
    features = grouped.agg(
        defect_count=(x_col, "size"),
        x_mean=(x_col, "mean"),
        x_std=(x_col, "std"),
        y_mean=(y_col, "mean"),
        y_std=(y_col, "std"),
        radius_mean=("radius_norm", "mean"),
        radius_std=("radius_norm", "std"),
        radius_max=("radius_norm", "max"),
    )
    features = features.fillna(0)

    zone_counts = pd.crosstab(enriched[wafer_id_col], enriched["radial_zone"])
    quadrant_counts = pd.crosstab(enriched[wafer_id_col], enriched["quadrant"])

    for zone in ["center", "middle", "edge"]:
        if zone not in zone_counts:
            zone_counts[zone] = 0
    for quadrant in ["Q1", "Q2", "Q3", "Q4"]:
        if quadrant not in quadrant_counts:
            quadrant_counts[quadrant] = 0

    zone_ratios = zone_counts[["center", "middle", "edge"]].div(zone_counts.sum(axis=1), axis=0)
    zone_ratios = zone_ratios.add_prefix("zone_ratio_")

    quadrant_ratios = quadrant_counts[["Q1", "Q2", "Q3", "Q4"]].div(quadrant_counts.sum(axis=1), axis=0)
    quadrant_ratios = quadrant_ratios.add_prefix("quadrant_ratio_")
    quadrant_imbalance = quadrant_ratios.max(axis=1) - quadrant_ratios.min(axis=1)
    quadrant_imbalance.name = "quadrant_imbalance"

    output = features.join(zone_ratios).join(quadrant_ratios).join(quadrant_imbalance)

    if defect_type_col and defect_type_col in enriched.columns:
        type_counts = pd.crosstab(enriched[wafer_id_col], enriched[defect_type_col])
        type_ratios = type_counts.div(type_counts.sum(axis=1), axis=0).add_prefix("defect_type_ratio_")
        output = output.join(type_ratios)

    return output.reset_index().rename(columns={wafer_id_col: "wafer_id"})


def heuristic_wafer_pattern_label(features: pd.DataFrame) -> pd.Series:
    """Assign a simple heuristic wafer defect pattern label from engineered features."""
    required = {"zone_ratio_center", "zone_ratio_edge", "quadrant_imbalance"}
    missing = sorted(required.difference(features.columns))
    if missing:
        raise ValueError(f"Missing wafer feature columns: {missing}")

    labels = np.select(
        [
            features["zone_ratio_edge"] >= 0.60,
            features["zone_ratio_center"] >= 0.60,
            features["quadrant_imbalance"] >= 0.50,
        ],
        ["edge", "center", "localized"],
        default="mixed",
    )
    return pd.Series(labels, index=features.index, name="pattern_label")
