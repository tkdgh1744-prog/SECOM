"""Training helpers for SECOM model selection and threshold tuning."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_MODEL_RANKING = ("fail_recall", "fail_f1", "pr_auc")
DEFAULT_THRESHOLD_RANKING = ("fail_f1", "fail_recall", "pr_auc")


@dataclass(frozen=True)
class SelectedResult:
    """Selected row metadata from a ranked model or threshold table."""

    row: dict[str, Any]
    ranking_columns: tuple[str, ...]

    @property
    def name(self) -> str | None:
        """Return the selected model name when present."""
        value = self.row.get("model")
        return str(value) if value is not None else None


def parse_thresholds(value: str | Iterable[float] | None) -> list[float]:
    """Parse comma-separated thresholds and validate they are probabilities."""
    if value is None:
        return [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

    if isinstance(value, str):
        raw_values = [item.strip() for item in value.split(",") if item.strip()]
        thresholds = [float(item) for item in raw_values]
    else:
        thresholds = [float(item) for item in value]

    invalid = [threshold for threshold in thresholds if threshold < 0 or threshold > 1]
    if invalid:
        raise ValueError(f"Thresholds must be between 0 and 1: {invalid}")
    if not thresholds:
        raise ValueError("At least one threshold is required.")
    return thresholds


def rank_results(
    results: pd.DataFrame,
    ranking_columns: Sequence[str] = DEFAULT_MODEL_RANKING,
) -> pd.DataFrame:
    """Sort a result table from best to worst using descending metric columns."""
    ranking_columns = tuple(ranking_columns)
    missing = [column for column in ranking_columns if column not in results.columns]
    if missing:
        raise ValueError(f"Missing ranking columns: {missing}")
    return results.sort_values(list(ranking_columns), ascending=False).reset_index(drop=True)


def select_top_result(
    results: pd.DataFrame,
    ranking_columns: Sequence[str] = DEFAULT_MODEL_RANKING,
) -> SelectedResult:
    """Return the highest ranked row from a model or threshold result table."""
    if results.empty:
        raise ValueError("Cannot select from an empty result table.")

    ranking_columns = tuple(ranking_columns)
    ranked = rank_results(results, ranking_columns=ranking_columns)
    return SelectedResult(row=ranked.iloc[0].to_dict(), ranking_columns=ranking_columns)


def build_threshold_metrics(
    y_true: pd.Series | np.ndarray,
    fail_probability: np.ndarray,
    thresholds: Iterable[float],
) -> pd.DataFrame:
    """Evaluate precision, recall, F1, and confusion counts for thresholds."""
    from sklearn.metrics import (
        average_precision_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    rows = []
    y_true_array = np.asarray(y_true)
    fail_probability = np.asarray(fail_probability, dtype=float)
    for threshold in parse_thresholds(thresholds):
        y_pred = (fail_probability >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true_array, y_pred, labels=[0, 1]).ravel()
        rows.append(
            {
                "threshold": threshold,
                "fail_precision": precision_score(
                    y_true_array, y_pred, pos_label=1, zero_division=0
                ),
                "fail_recall": recall_score(y_true_array, y_pred, pos_label=1, zero_division=0),
                "fail_f1": f1_score(y_true_array, y_pred, pos_label=1, zero_division=0),
                "roc_auc": roc_auc_score(y_true_array, fail_probability),
                "pr_auc": average_precision_score(y_true_array, fail_probability),
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "tp": tp,
            }
        )
    return pd.DataFrame(rows)
