"""Reusable data contract checks for manufacturing analytics tables."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class ContractResult:
    """Validation result for a data contract."""

    table_name: str
    errors: tuple[str, ...]

    @property
    def ok(self) -> bool:
        """Return True when no contract errors were found."""
        return not self.errors

    def raise_if_failed(self) -> None:
        """Raise ValueError when validation failed."""
        if self.errors:
            joined = "; ".join(self.errors)
            raise ValueError(f"{self.table_name} contract failed: {joined}")


def _missing_columns(df: pd.DataFrame, columns: Iterable[str]) -> list[str]:
    return [column for column in columns if column not in df.columns]


def require_columns(df: pd.DataFrame, columns: Iterable[str], table_name: str) -> ContractResult:
    """Validate required columns."""
    missing = _missing_columns(df, columns)
    errors = tuple(f"missing columns: {missing}" for _ in [0] if missing)
    return ContractResult(table_name=table_name, errors=errors)


def require_numeric_columns(df: pd.DataFrame, columns: Iterable[str], table_name: str) -> ContractResult:
    """Validate that columns exist and are numeric."""
    errors: list[str] = []
    missing = _missing_columns(df, columns)
    if missing:
        errors.append(f"missing numeric columns: {missing}")

    for column in columns:
        if column in df.columns and not pd.api.types.is_numeric_dtype(df[column]):
            errors.append(f"non-numeric column: {column}")

    return ContractResult(table_name=table_name, errors=tuple(errors))


def require_value_set(
    df: pd.DataFrame,
    column: str,
    allowed_values: set,
    table_name: str,
    allow_missing: bool = False,
) -> ContractResult:
    """Validate that a column contains only allowed values."""
    if column not in df.columns:
        return ContractResult(table_name=table_name, errors=(f"missing column: {column}",))

    values = df[column]
    if allow_missing:
        values = values.dropna()

    invalid = sorted(set(values).difference(allowed_values))
    errors = tuple(f"invalid values in {column}: {invalid}" for _ in [0] if invalid)
    return ContractResult(table_name=table_name, errors=errors)


def require_probability_column(df: pd.DataFrame, column: str, table_name: str) -> ContractResult:
    """Validate that a probability column exists, is numeric, and lies in [0, 1]."""
    numeric_result = require_numeric_columns(df, [column], table_name)
    errors = list(numeric_result.errors)
    if column in df.columns and pd.api.types.is_numeric_dtype(df[column]):
        invalid = df[column].dropna().lt(0).any() or df[column].dropna().gt(1).any()
        if invalid:
            errors.append(f"probability out of range [0, 1]: {column}")
    return ContractResult(table_name=table_name, errors=tuple(errors))


def merge_contract_results(table_name: str, results: Iterable[ContractResult]) -> ContractResult:
    """Merge multiple contract results into one."""
    errors: list[str] = []
    for result in results:
        errors.extend(result.errors)
    return ContractResult(table_name=table_name, errors=tuple(errors))


def validate_wafer_inspection_contract(df: pd.DataFrame) -> ContractResult:
    """Validate the expected wafer inspection coordinate table."""
    return merge_contract_results(
        "wafer_inspection",
        [
            require_columns(df, ["wafer_id", "x", "y"], "wafer_inspection"),
            require_numeric_columns(df, ["x", "y"], "wafer_inspection"),
        ],
    )


def validate_equipment_events_contract(df: pd.DataFrame) -> ContractResult:
    """Validate the expected equipment event table."""
    return require_columns(
        df,
        ["equipment_id", "timestamp", "event_type"],
        "equipment_events",
    )


def validate_predictions_contract(df: pd.DataFrame) -> ContractResult:
    """Validate prediction output used by monitoring reports."""
    return merge_contract_results(
        "predictions",
        [
            require_columns(df, ["prediction", "fail_probability"], "predictions"),
            require_value_set(df, "prediction", {"Pass", "Fail"}, "predictions"),
            require_probability_column(df, "fail_probability", "predictions"),
        ],
    )


def validate_modeling_table_contract(df: pd.DataFrame, feature_columns: list[str]) -> ContractResult:
    """Validate that a modeling table contains required model feature columns."""
    return require_columns(df, feature_columns, "modeling_table")
