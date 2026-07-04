"""Monitoring reports for prediction outputs and process quality risk."""

from __future__ import annotations

import pandas as pd


REQUIRED_PREDICTION_COLUMNS = {"prediction", "fail_probability"}


def validate_prediction_columns(predictions: pd.DataFrame) -> None:
    """Validate required prediction output columns."""
    missing = sorted(REQUIRED_PREDICTION_COLUMNS.difference(predictions.columns))
    if missing:
        raise ValueError(f"Missing prediction columns: {missing}")


def overall_risk_summary(
    predictions: pd.DataFrame,
    probability_col: str = "fail_probability",
    prediction_col: str = "prediction",
    high_risk_threshold: float = 0.5,
) -> pd.Series:
    """Return an overall process-quality risk summary from prediction rows."""
    validate_prediction_columns(predictions)
    fail_probability = predictions[probability_col]
    fail_predictions = predictions[prediction_col].astype(str).str.lower().eq("fail")
    high_risk = fail_probability >= high_risk_threshold

    return pd.Series(
        {
            "n_predictions": len(predictions),
            "mean_fail_probability": float(fail_probability.mean()),
            "max_fail_probability": float(fail_probability.max()),
            "predicted_fail_count": int(fail_predictions.sum()),
            "predicted_fail_ratio": float(fail_predictions.mean()),
            "high_risk_count": int(high_risk.sum()),
            "high_risk_ratio": float(high_risk.mean()),
            "high_risk_threshold": high_risk_threshold,
        },
        name="value",
    )


def group_risk_summary(
    predictions: pd.DataFrame,
    group_cols: list[str],
    probability_col: str = "fail_probability",
    prediction_col: str = "prediction",
    high_risk_threshold: float = 0.5,
    alert_ratio_threshold: float = 0.25,
) -> pd.DataFrame:
    """Aggregate prediction risk by wafer, equipment, lot, or other group columns."""
    validate_prediction_columns(predictions)
    missing_groups = [column for column in group_cols if column not in predictions.columns]
    if missing_groups:
        raise ValueError(f"Missing group columns: {missing_groups}")

    data = predictions.copy()
    data["_predicted_fail"] = data[prediction_col].astype(str).str.lower().eq("fail")
    data["_high_risk"] = data[probability_col] >= high_risk_threshold

    summary = (
        data.groupby(group_cols, dropna=False)
        .agg(
            n_predictions=(probability_col, "size"),
            mean_fail_probability=(probability_col, "mean"),
            max_fail_probability=(probability_col, "max"),
            predicted_fail_count=("_predicted_fail", "sum"),
            high_risk_count=("_high_risk", "sum"),
        )
        .reset_index()
    )
    summary["predicted_fail_ratio"] = summary["predicted_fail_count"] / summary["n_predictions"]
    summary["high_risk_ratio"] = summary["high_risk_count"] / summary["n_predictions"]
    summary["alert_flag"] = summary["high_risk_ratio"] >= alert_ratio_threshold

    return summary.sort_values(
        ["alert_flag", "high_risk_ratio", "mean_fail_probability"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def top_risk_predictions(
    predictions: pd.DataFrame,
    top_n: int = 20,
    probability_col: str = "fail_probability",
) -> pd.DataFrame:
    """Return the highest-risk prediction rows."""
    validate_prediction_columns(predictions)
    return predictions.sort_values(probability_col, ascending=False).head(top_n).reset_index(drop=True)
