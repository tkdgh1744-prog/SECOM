"""Run the SECOM manufacturing analytics pipeline end to end."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from typing import Callable

from scripts.assemble_feature_table import assemble_from_paths
from scripts.generate_monitoring_report import generate_monitoring_reports
from scripts.predict_with_model import run_prediction
from scripts.run_quality_report import run_quality_report


@dataclass(frozen=True)
class PipelinePaths:
    """Input and output paths for an end-to-end pipeline run."""

    raw_data_dir: Path
    sensor_path: Path
    wafer_path: Path | None
    equipment_path: Path | None
    model_path: Path | None
    reports_dir: Path
    features_path: Path
    predictions_path: Path
    monitoring_dir: Path


@dataclass(frozen=True)
class PipelineResult:
    """Output artifact paths from a pipeline run."""

    quality_report_dir: Path
    feature_table_path: Path
    join_report_path: Path
    feature_missingness_path: Path
    predictions_path: Path | None
    monitoring_paths: dict[str, Path]


def parse_optional_path(value: str | None) -> Path | None:
    """Parse optional path arguments."""
    if value is None or not str(value).strip():
        return None
    return Path(value)


def parse_columns(value: str | None) -> list[str]:
    """Parse optional comma-separated columns."""
    if value is None or not value.strip():
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the manufacturing analytics pipeline.")
    parser.add_argument("--raw-data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--sensor-path", type=Path, required=True)
    parser.add_argument("--wafer-path", type=parse_optional_path, default=None)
    parser.add_argument("--equipment-path", type=parse_optional_path, default=None)
    parser.add_argument("--model-path", type=parse_optional_path, default=None)
    parser.add_argument("--reports-dir", type=Path, default=Path("outputs/reports"))
    parser.add_argument("--features-path", type=Path, default=Path("outputs/features/modeling_table.csv"))
    parser.add_argument("--predictions-path", type=Path, default=Path("outputs/predictions/predictions.csv"))
    parser.add_argument("--monitoring-dir", type=Path, default=Path("outputs/reports/monitoring"))
    parser.add_argument("--download-secom", action="store_true")
    parser.add_argument("--skip-quality-report", action="store_true")
    parser.add_argument("--skip-prediction", action="store_true")
    parser.add_argument("--id-columns", type=parse_columns, default=[])
    parser.add_argument("--monitoring-group-columns", type=parse_columns, default=[])
    return parser.parse_args()


def run_pipeline(
    paths: PipelinePaths,
    download_secom: bool = False,
    skip_quality_report: bool = False,
    skip_prediction: bool = False,
    id_columns: list[str] | None = None,
    monitoring_group_columns: list[str] | None = None,
    quality_step: Callable = run_quality_report,
    assemble_step: Callable = assemble_from_paths,
    prediction_step: Callable = run_prediction,
    monitoring_step: Callable = generate_monitoring_reports,
) -> PipelineResult:
    """Run quality, feature assembly, prediction, and monitoring steps."""
    quality_report_dir = paths.reports_dir / "quality"
    if not skip_quality_report:
        quality_step(
            raw_data_dir=paths.raw_data_dir,
            output_dir=quality_report_dir,
            download=download_secom,
        )

    feature_table_path, join_report_path, feature_missingness_path = assemble_step(
        sensor_path=paths.sensor_path,
        wafer_path=paths.wafer_path,
        equipment_path=paths.equipment_path,
        output_path=paths.features_path,
        report_dir=paths.reports_dir,
    )

    predictions_path = None
    monitoring_paths: dict[str, Path] = {}
    should_predict = not skip_prediction and paths.model_path is not None
    if should_predict:
        predictions_path = prediction_step(
            model_path=paths.model_path,
            features_path=feature_table_path,
            output_path=paths.predictions_path,
            id_columns=id_columns or [],
        )
        monitoring_paths = monitoring_step(
            predictions_path=predictions_path,
            output_dir=paths.monitoring_dir,
            group_columns=monitoring_group_columns or [],
        )

    return PipelineResult(
        quality_report_dir=quality_report_dir,
        feature_table_path=feature_table_path,
        join_report_path=join_report_path,
        feature_missingness_path=feature_missingness_path,
        predictions_path=predictions_path,
        monitoring_paths=monitoring_paths,
    )


def main() -> int:
    args = parse_args()
    result = run_pipeline(
        paths=PipelinePaths(
            raw_data_dir=args.raw_data_dir,
            sensor_path=args.sensor_path,
            wafer_path=args.wafer_path,
            equipment_path=args.equipment_path,
            model_path=args.model_path,
            reports_dir=args.reports_dir,
            features_path=args.features_path,
            predictions_path=args.predictions_path,
            monitoring_dir=args.monitoring_dir,
        ),
        download_secom=args.download_secom,
        skip_quality_report=args.skip_quality_report,
        skip_prediction=args.skip_prediction,
        id_columns=args.id_columns,
        monitoring_group_columns=args.monitoring_group_columns,
    )
    print(f"Feature table: {result.feature_table_path}")
    print(f"Join report: {result.join_report_path}")
    print(f"Feature missingness report: {result.feature_missingness_path}")
    if result.predictions_path:
        print(f"Predictions: {result.predictions_path}")
    for name, path in result.monitoring_paths.items():
        print(f"Monitoring {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
