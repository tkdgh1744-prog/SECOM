"""Train and save a SECOM Pass/Fail classification model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from typing import Any

import pandas as pd

from src.model_registry import ModelBundle, save_model_bundle
from src.secom_data import default_secom_paths, download_secom_dataset, load_secom_data
from src.secom_modeling import (
    HighMissingFeatureDropper,
    evaluate_classifier,
    get_positive_proba,
    make_linear_pipeline,
    make_tree_pipeline,
    predict_with_threshold,
)
from src.secom_training import (
    DEFAULT_MODEL_RANKING,
    DEFAULT_THRESHOLD_RANKING,
    build_threshold_metrics,
    parse_thresholds,
    rank_results,
    select_top_result,
)


def parse_optional_path(value: str | None) -> Path | None:
    """Parse optional path arguments."""
    if value is None or not str(value).strip():
        return None
    return Path(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a SECOM Pass/Fail model.")
    parser.add_argument("--raw-data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--feature-path", type=parse_optional_path, default=None)
    parser.add_argument("--label-path", type=parse_optional_path, default=None)
    parser.add_argument("--download-secom", action="store_true")
    parser.add_argument("--model-output", type=Path, default=Path("outputs/models/secom_final_pipeline.joblib"))
    parser.add_argument("--metrics-output", type=Path, default=Path("outputs/reports/model_metrics.csv"))
    parser.add_argument("--threshold-output", type=Path, default=Path("outputs/reports/threshold_metrics.csv"))
    parser.add_argument("--predictions-output", type=Path, default=Path("outputs/predictions/test_predictions.csv"))
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--validation-size", type=float, default=0.25)
    parser.add_argument("--missing-threshold", type=float, default=0.5)
    parser.add_argument("--low-variance-threshold", type=float, default=1e-8)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--thresholds", type=parse_thresholds, default=parse_thresholds(None))
    return parser.parse_args()


def resolve_secom_paths(
    raw_data_dir: Path,
    feature_path: Path | None = None,
    label_path: Path | None = None,
    download: bool = False,
) -> tuple[Path, Path]:
    """Resolve SECOM feature and label paths, optionally downloading them."""
    if download:
        downloaded_feature_path, downloaded_label_path = download_secom_dataset(raw_data_dir)
        feature_path = feature_path or downloaded_feature_path
        label_path = label_path or downloaded_label_path
    else:
        default_feature_path, default_label_path = default_secom_paths(raw_data_dir)
        feature_path = feature_path or default_feature_path
        label_path = label_path or default_label_path

    missing = [path for path in [feature_path, label_path] if path is None or not Path(path).exists()]
    if missing:
        raise FileNotFoundError(
            "SECOM files are missing. Pass --download-secom or provide --feature-path and --label-path."
        )
    return Path(feature_path), Path(label_path)


def make_candidate_models(
    missing_threshold: float = 0.5,
    low_variance_threshold: float = 1e-8,
    random_state: int = 42,
) -> dict[str, tuple[str, Any]]:
    """Create candidate models for baseline, linear, and tree-based comparison."""
    from sklearn.dummy import DummyClassifier
    from sklearn.feature_selection import VarianceThreshold
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline

    dummy_pipeline = Pipeline(
        steps=[
            ("drop_high_missing", HighMissingFeatureDropper(threshold=missing_threshold)),
            ("imputer", SimpleImputer(strategy="median")),
            ("variance", VarianceThreshold(threshold=low_variance_threshold)),
            ("model", DummyClassifier(strategy="most_frequent", random_state=random_state)),
        ]
    )
    return {
        "Dummy Most Frequent": ("baseline", dummy_pipeline),
        "Logistic Regression": (
            "none",
            make_linear_pipeline(
                missing_threshold=missing_threshold,
                low_variance_threshold=low_variance_threshold,
                class_weight=None,
                random_state=random_state,
            ),
        ),
        "Logistic Regression Balanced": (
            "class_weight=balanced",
            make_linear_pipeline(
                missing_threshold=missing_threshold,
                low_variance_threshold=low_variance_threshold,
                class_weight="balanced",
                random_state=random_state,
            ),
        ),
        "Random Forest Balanced": (
            "class_weight=balanced",
            make_tree_pipeline(
                missing_threshold=missing_threshold,
                low_variance_threshold=low_variance_threshold,
                class_weight="balanced",
                random_state=random_state,
            ),
        ),
    }


def write_test_predictions(
    output_path: Path,
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    timestamps: pd.Series,
    threshold: float,
) -> Path:
    """Write held-out test predictions with labels and timestamps."""
    prediction_frame = predict_with_threshold(model, X_test, threshold=threshold).to_frame(index=X_test.index)
    output = pd.DataFrame(
        {
            "sample_index": X_test.index,
            "timestamp": timestamps.loc[X_test.index].astype(str).to_numpy(),
            "actual_label": y_test.to_numpy(),
            "actual_name": y_test.replace({0: "Pass", 1: "Fail"}).to_numpy(),
        },
        index=X_test.index,
    )
    output = pd.concat([output, prediction_frame], axis=1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    return output_path


def train_secom_model(
    raw_data_dir: Path = Path("data/raw"),
    feature_path: Path | None = None,
    label_path: Path | None = None,
    download_secom: bool = False,
    model_output: Path = Path("outputs/models/secom_final_pipeline.joblib"),
    metrics_output: Path = Path("outputs/reports/model_metrics.csv"),
    threshold_output: Path = Path("outputs/reports/threshold_metrics.csv"),
    predictions_output: Path = Path("outputs/predictions/test_predictions.csv"),
    test_size: float = 0.2,
    validation_size: float = 0.25,
    missing_threshold: float = 0.5,
    low_variance_threshold: float = 1e-8,
    random_state: int = 42,
    thresholds: list[float] | None = None,
) -> dict[str, Path]:
    """Train candidate models, tune threshold, and save final artifacts."""
    from sklearn.model_selection import train_test_split

    feature_path, label_path = resolve_secom_paths(
        raw_data_dir=raw_data_dir,
        feature_path=feature_path,
        label_path=label_path,
        download=download_secom,
    )
    dataset = load_secom_data(feature_path, label_path)
    thresholds = thresholds or parse_thresholds(None)

    X_train_valid, X_test, y_train_valid, y_test = train_test_split(
        dataset.features,
        dataset.labels,
        test_size=test_size,
        stratify=dataset.labels,
        random_state=random_state,
    )
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_valid,
        y_train_valid,
        test_size=validation_size,
        stratify=y_train_valid,
        random_state=random_state,
    )

    validation_rows = []
    candidates = make_candidate_models(
        missing_threshold=missing_threshold,
        low_variance_threshold=low_variance_threshold,
        random_state=random_state,
    )
    for model_name, (imbalance_method, model) in candidates.items():
        model.fit(X_train, y_train)
        validation_rows.append(
            {
                "split": "validation",
                **evaluate_classifier(
                    model_name,
                    imbalance_method,
                    model,
                    X_valid,
                    y_valid,
                    threshold=0.5,
                ),
            }
        )

    validation_metrics = rank_results(pd.DataFrame(validation_rows), ranking_columns=DEFAULT_MODEL_RANKING)
    selected_model_name = str(select_top_result(validation_metrics, DEFAULT_MODEL_RANKING).row["model"])

    selected_candidates = make_candidate_models(
        missing_threshold=missing_threshold,
        low_variance_threshold=low_variance_threshold,
        random_state=random_state,
    )
    selected_imbalance_method, threshold_model = selected_candidates[selected_model_name]
    threshold_model.fit(X_train, y_train)
    validation_probability = get_positive_proba(threshold_model, X_valid)
    threshold_metrics = rank_results(
        build_threshold_metrics(y_valid, validation_probability, thresholds),
        ranking_columns=DEFAULT_THRESHOLD_RANKING,
    )
    selected_threshold = float(
        select_top_result(threshold_metrics, DEFAULT_THRESHOLD_RANKING).row["threshold"]
    )

    final_candidates = make_candidate_models(
        missing_threshold=missing_threshold,
        low_variance_threshold=low_variance_threshold,
        random_state=random_state,
    )
    _, final_model = final_candidates[selected_model_name]
    final_model.fit(X_train_valid, y_train_valid)

    test_metrics = pd.DataFrame(
        [
            {
                "split": "test",
                **evaluate_classifier(
                    selected_model_name,
                    selected_imbalance_method,
                    final_model,
                    X_test,
                    y_test,
                    threshold=selected_threshold,
                ),
            }
        ]
    )
    metrics = pd.concat([validation_metrics, test_metrics], ignore_index=True)

    bundle = ModelBundle(
        model=final_model,
        threshold=selected_threshold,
        feature_columns=dataset.features.columns.tolist(),
        target_mapping={0: "Pass", 1: "Fail"},
        model_name=selected_model_name,
        metrics=test_metrics.iloc[0].to_dict(),
    )

    metrics_output.parent.mkdir(parents=True, exist_ok=True)
    threshold_output.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(metrics_output, index=False)
    threshold_metrics.to_csv(threshold_output, index=False)
    save_model_bundle(bundle, model_output)
    write_test_predictions(
        output_path=predictions_output,
        model=final_model,
        X_test=X_test,
        y_test=y_test,
        timestamps=dataset.timestamps,
        threshold=selected_threshold,
    )

    return {
        "model": model_output,
        "metrics": metrics_output,
        "thresholds": threshold_output,
        "test_predictions": predictions_output,
    }


def main() -> int:
    args = parse_args()
    outputs = train_secom_model(
        raw_data_dir=args.raw_data_dir,
        feature_path=args.feature_path,
        label_path=args.label_path,
        download_secom=args.download_secom,
        model_output=args.model_output,
        metrics_output=args.metrics_output,
        threshold_output=args.threshold_output,
        predictions_output=args.predictions_output,
        test_size=args.test_size,
        validation_size=args.validation_size,
        missing_threshold=args.missing_threshold,
        low_variance_threshold=args.low_variance_threshold,
        random_state=args.random_state,
        thresholds=args.thresholds,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
