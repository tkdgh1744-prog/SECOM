"""Generate SECOM data quality reports from local or downloaded raw files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.quality_reports import quality_report_bundle
from src.secom_data import default_secom_paths, download_secom_dataset, load_secom_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate SECOM data quality reports.")
    parser.add_argument(
        "--raw-data-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory containing secom.data and secom_labels.data.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/reports"),
        help="Directory where report CSV files will be written.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download UCI SECOM raw files before loading.",
    )
    parser.add_argument(
        "--missing-threshold",
        type=float,
        default=0.5,
        help="Missing ratio threshold for high-missing feature reporting.",
    )
    return parser.parse_args()


def write_report_bundle(report_bundle: dict, output_dir: Path) -> None:
    """Write quality report bundle objects to CSV files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    overview = report_bundle["overview"].to_frame().reset_index()
    overview.columns = ["metric", "value"]
    overview.to_csv(output_dir / "overview.csv", index=False)

    report_bundle["class_distribution"].to_csv(output_dir / "class_distribution.csv", index=False)
    report_bundle["missingness"].to_csv(output_dir / "missingness.csv", index_label="feature")

    pd.Series(report_bundle["high_missing_features"], name="feature").to_csv(
        output_dir / "high_missing_features.csv",
        index=False,
    )
    pd.Series(report_bundle["constant_features"], name="feature").to_csv(
        output_dir / "constant_features.csv",
        index=False,
    )


def run_quality_report(
    raw_data_dir: Path,
    output_dir: Path,
    download: bool = False,
    missing_threshold: float = 0.5,
) -> Path:
    """Load SECOM data and write quality reports to an output directory."""
    if download:
        feature_path, label_path = download_secom_dataset(raw_data_dir)
    else:
        feature_path, label_path = default_secom_paths(raw_data_dir)

    dataset = load_secom_data(feature_path, label_path)
    report_bundle = quality_report_bundle(dataset, missing_threshold=missing_threshold)
    write_report_bundle(report_bundle, output_dir)
    return output_dir


def main() -> int:
    args = parse_args()
    output_dir = run_quality_report(
        raw_data_dir=args.raw_data_dir,
        output_dir=args.output_dir,
        download=args.download,
        missing_threshold=args.missing_threshold,
    )
    print(f"Quality reports written to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
