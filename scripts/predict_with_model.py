"""Run predictions with a saved SECOM model bundle and feature CSV."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.model_registry import load_model_bundle, predict_from_bundle


def parse_id_columns(value: str | None) -> list[str]:
    """Parse optional comma-separated identifier columns."""
    if value is None or not value.strip():
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict Pass/Fail using a saved model bundle.")
    parser.add_argument("--model-path", type=Path, required=True, help="Path to a saved model bundle.")
    parser.add_argument("--features-path", type=Path, required=True, help="CSV file containing feature rows.")
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("outputs/predictions/predictions.csv"),
        help="Output CSV path for predictions.",
    )
    parser.add_argument(
        "--id-columns",
        type=parse_id_columns,
        default=[],
        help="Optional comma-separated identifier columns to include in prediction output.",
    )
    return parser.parse_args()


def run_prediction(
    model_path: Path,
    features_path: Path,
    output_path: Path,
    id_columns: list[str] | None = None,
) -> Path:
    """Load a model bundle and feature table, then write prediction results."""
    if not features_path.exists():
        raise FileNotFoundError(f"Feature CSV not found: {features_path}")

    bundle = load_model_bundle(model_path)
    feature_table = pd.read_csv(features_path)
    predictions = predict_from_bundle(feature_table, bundle)

    id_columns = id_columns or []
    missing_id_columns = [column for column in id_columns if column not in feature_table.columns]
    if missing_id_columns:
        raise ValueError(f"Missing requested id columns: {missing_id_columns}")

    if id_columns:
        predictions = pd.concat([feature_table.loc[:, id_columns], predictions], axis=1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path, index=False)
    return output_path


def main() -> int:
    args = parse_args()
    output_path = run_prediction(
        model_path=args.model_path,
        features_path=args.features_path,
        output_path=args.output_path,
        id_columns=args.id_columns,
    )
    print(f"Predictions written to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
