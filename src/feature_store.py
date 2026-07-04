"""Utilities for assembling sensor, wafer, and equipment feature tables."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class JoinReport:
    """Summary of a feature-table join operation."""

    table_name: str
    join_key: str
    left_rows: int
    right_rows: int
    output_rows: int
    unmatched_rows: int

    def to_dict(self) -> dict[str, int | str]:
        """Return a dictionary representation for reporting."""
        return {
            "table_name": self.table_name,
            "join_key": self.join_key,
            "left_rows": self.left_rows,
            "right_rows": self.right_rows,
            "output_rows": self.output_rows,
            "unmatched_rows": self.unmatched_rows,
        }


def validate_key_columns(df: pd.DataFrame, keys: list[str], table_name: str) -> None:
    """Validate that all key columns exist in a table."""
    missing = [key for key in keys if key not in df.columns]
    if missing:
        raise ValueError(f"{table_name} is missing key columns: {missing}")


def prefix_feature_columns(df: pd.DataFrame, keys: list[str], prefix: str) -> pd.DataFrame:
    """Prefix non-key feature columns to avoid collisions after joins."""
    rename_map = {column: f"{prefix}{column}" for column in df.columns if column not in keys}
    return df.rename(columns=rename_map)


def left_join_features(
    base_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    keys: list[str],
    table_name: str,
    feature_prefix: str,
) -> tuple[pd.DataFrame, JoinReport]:
    """Left join one feature table onto a base table and return a join report."""
    validate_key_columns(base_df, keys, "base_df")
    validate_key_columns(feature_df, keys, table_name)

    right = prefix_feature_columns(feature_df, keys=keys, prefix=feature_prefix)
    before_cols = set(base_df.columns)
    output = base_df.merge(right, on=keys, how="left", indicator=f"_{table_name}_merge")
    unmatched = int((output[f"_{table_name}_merge"] == "left_only").sum())
    output = output.drop(columns=[f"_{table_name}_merge"])

    report = JoinReport(
        table_name=table_name,
        join_key=",".join(keys),
        left_rows=len(base_df),
        right_rows=len(feature_df),
        output_rows=len(output),
        unmatched_rows=unmatched,
    )

    duplicated_columns = [column for column in output.columns if column in before_cols and column not in base_df.columns]
    if duplicated_columns:
        raise ValueError(f"Unexpected duplicated columns after join: {duplicated_columns}")

    return output, report


def assemble_feature_table(
    sensor_df: pd.DataFrame,
    wafer_features: pd.DataFrame | None = None,
    equipment_features: pd.DataFrame | None = None,
    sensor_wafer_keys: list[str] | None = None,
    sensor_equipment_keys: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assemble sensor, wafer, and equipment features into one modeling table."""
    output = sensor_df.copy()
    reports: list[JoinReport] = []

    if wafer_features is not None:
        keys = sensor_wafer_keys or ["wafer_id"]
        output, report = left_join_features(
            output,
            wafer_features,
            keys=keys,
            table_name="wafer_features",
            feature_prefix="wafer_",
        )
        reports.append(report)

    if equipment_features is not None:
        keys = sensor_equipment_keys or ["equipment_id"]
        output, report = left_join_features(
            output,
            equipment_features,
            keys=keys,
            table_name="equipment_features",
            feature_prefix="equipment_",
        )
        reports.append(report)

    return output, pd.DataFrame([report.to_dict() for report in reports])


def feature_missingness_report(feature_table: pd.DataFrame) -> pd.DataFrame:
    """Return missing-value counts and ratios for a modeling feature table."""
    report = pd.DataFrame(
        {
            "missing_count": feature_table.isna().sum(),
            "missing_ratio": feature_table.isna().mean(),
        }
    )
    return report.sort_values(["missing_ratio", "missing_count"], ascending=False)
