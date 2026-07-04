"""Assemble sensor, wafer, and equipment features into a modeling table."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.feature_store import assemble_feature_table, feature_missingness_report


def parse_key_list(value: str) -> list[str]:
    """Parse a comma-separated key list."""
    keys = [item.strip() for item in value.split(",") if item.strip()]
    if not keys:
        raise argparse.ArgumentTypeError("At least one key column is required.")
    return keys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assemble manufacturing feature tables.")
    parser.add_argument("--sensor-path", type=Path, required=True, help="CSV file containing sensor features.")
    parser.add_argument("--wafer-path", type=Path, default=None, help="Optional wafer feature CSV file.")
    parser.add_argument("--equipment-path", type=Path, default=None, help="Optional equipment feature CSV file.")
    parser.add_argument(
        "--sensor-wafer-keys",
        type=parse_key_list,
        default=["wafer_id"],
        help="Comma-separated join keys between sensor and wafer features.",
    )
    parser.add_argument(
        "--sensor-equipment-keys",
        type=parse_key_list,
        default=["equipment_id"],
        help="Comma-separated join keys between sensor and equipment features.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("outputs/features/modeling_table.csv"),
        help="Output CSV path for assembled modeling table.",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("outputs/reports"),
        help="Directory where join and missingness reports will be written.",
    )
    return parser.parse_args()


def read_optional_csv(path: Path | None) -> pd.DataFrame | None:
    """Read an optional CSV file."""
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(f"Optional feature file not found: {path}")
    return pd.read_csv(path)


def assemble_from_paths(
    sensor_path: Path,
    output_path: Path,
    report_dir: Path,
    wafer_path: Path | None = None,
    equipment_path: Path | None = None,
    sensor_wafer_keys: list[str] | None = None,
    sensor_equipment_keys: list[str] | None = None,
) -> tuple[Path, Path, Path]:
    """Assemble feature tables from CSV paths and write output files."""
    if not sensor_path.exists():
        raise FileNotFoundError(f"Sensor feature file not found: {sensor_path}")

    sensor_df = pd.read_csv(sensor_path)
    wafer_df = read_optional_csv(wafer_path)
    equipment_df = read_optional_csv(equipment_path)

    feature_table, join_report = assemble_feature_table(
        sensor_df=sensor_df,
        wafer_features=wafer_df,
        equipment_features=equipment_df,
        sensor_wafer_keys=sensor_wafer_keys,
        sensor_equipment_keys=sensor_equipment_keys,
    )
    missingness = feature_missingness_report(feature_table)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    join_report_path = report_dir / "feature_join_report.csv"
    missingness_path = report_dir / "feature_missingness_report.csv"

    feature_table.to_csv(output_path, index=False)
    join_report.to_csv(join_report_path, index=False)
    missingness.to_csv(missingness_path, index_label="feature")

    return output_path, join_report_path, missingness_path


def main() -> int:
    args = parse_args()
    output_path, join_report_path, missingness_path = assemble_from_paths(
        sensor_path=args.sensor_path,
        wafer_path=args.wafer_path,
        equipment_path=args.equipment_path,
        sensor_wafer_keys=args.sensor_wafer_keys,
        sensor_equipment_keys=args.sensor_equipment_keys,
        output_path=args.output_path,
        report_dir=args.report_dir,
    )
    print(f"Modeling table written to: {output_path}")
    print(f"Join report written to: {join_report_path}")
    print(f"Missingness report written to: {missingness_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
