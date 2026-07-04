"""Feature engineering utilities for equipment event and failure data."""

from __future__ import annotations

import pandas as pd


def validate_equipment_columns(
    events_df: pd.DataFrame,
    equipment_id_col: str = "equipment_id",
    timestamp_col: str = "timestamp",
    event_type_col: str = "event_type",
) -> None:
    """Validate that an equipment event table has required columns."""
    required = {equipment_id_col, timestamp_col, event_type_col}
    missing = sorted(required.difference(events_df.columns))
    if missing:
        raise ValueError(f"Missing equipment event columns: {missing}")


def prepare_equipment_events(
    events_df: pd.DataFrame,
    equipment_id_col: str = "equipment_id",
    timestamp_col: str = "timestamp",
    event_type_col: str = "event_type",
) -> pd.DataFrame:
    """Return events sorted by equipment and timestamp with parsed datetimes."""
    validate_equipment_columns(
        events_df,
        equipment_id_col=equipment_id_col,
        timestamp_col=timestamp_col,
        event_type_col=event_type_col,
    )
    result = events_df.copy()
    result[timestamp_col] = pd.to_datetime(result[timestamp_col], errors="coerce")
    if result[timestamp_col].isna().any():
        raise ValueError("Equipment event timestamps contain unparsable values.")
    return result.sort_values([equipment_id_col, timestamp_col]).reset_index(drop=True)


def equipment_event_features(
    events_df: pd.DataFrame,
    equipment_id_col: str = "equipment_id",
    timestamp_col: str = "timestamp",
    event_type_col: str = "event_type",
    failure_label_col: str | None = "failure_label",
) -> pd.DataFrame:
    """Aggregate equipment event logs into equipment-level predictive features."""
    prepared = prepare_equipment_events(
        events_df,
        equipment_id_col=equipment_id_col,
        timestamp_col=timestamp_col,
        event_type_col=event_type_col,
    )

    grouped = prepared.groupby(equipment_id_col, dropna=False)
    features = grouped.agg(
        event_count=(event_type_col, "size"),
        first_event_time=(timestamp_col, "min"),
        last_event_time=(timestamp_col, "max"),
        unique_event_types=(event_type_col, "nunique"),
    )
    features["observation_hours"] = (
        features["last_event_time"] - features["first_event_time"]
    ).dt.total_seconds() / 3600.0
    features["event_rate_per_hour"] = features["event_count"] / features["observation_hours"].clip(lower=1)

    event_type_counts = pd.crosstab(prepared[equipment_id_col], prepared[event_type_col])
    event_type_counts = event_type_counts.add_prefix("event_type_count_")
    output = features.join(event_type_counts).fillna(0)

    if failure_label_col and failure_label_col in prepared.columns:
        failure = prepared.groupby(equipment_id_col)[failure_label_col].max().rename("failure_label")
        output = output.join(failure)

    return output.reset_index().rename(columns={equipment_id_col: "equipment_id"})


def add_time_since_previous_event(
    events_df: pd.DataFrame,
    equipment_id_col: str = "equipment_id",
    timestamp_col: str = "timestamp",
    event_type_col: str = "event_type",
) -> pd.DataFrame:
    """Add elapsed hours since the previous event for each equipment unit."""
    prepared = prepare_equipment_events(
        events_df,
        equipment_id_col=equipment_id_col,
        timestamp_col=timestamp_col,
        event_type_col=event_type_col,
    )
    elapsed = prepared.groupby(equipment_id_col)[timestamp_col].diff().dt.total_seconds() / 3600.0
    prepared["hours_since_previous_event"] = elapsed.fillna(0)
    return prepared
