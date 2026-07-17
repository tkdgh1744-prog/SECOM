"""Validate a trusted cloud-mounted WM-811K pickle before training."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.wafer_dataset import inspect_wm811k_dataset, write_wm811k_preflight


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than 0")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check WM-811K structure and provenance before analysis or training."
    )
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("outputs/wafer_maps/wm811k_preflight.json"),
    )
    parser.add_argument("--wafer-map-col", default="waferMap")
    parser.add_argument("--label-col", default="failureType")
    parser.add_argument("--group-col", default="lotName")
    parser.add_argument("--id-col", default=None)
    parser.add_argument("--sample-records", type=positive_int, default=5000)
    parser.add_argument("--integration-mode", choices=["real", "synthetic"], default="real")
    parser.add_argument("--source-uri", default=None)
    parser.add_argument("--license-note", default=None)
    parser.add_argument("--compute-sha256", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = inspect_wm811k_dataset(
        args.input_path,
        wafer_map_col=args.wafer_map_col,
        label_col=args.label_col,
        group_col=args.group_col,
        id_col=args.id_col,
        sample_records=args.sample_records,
        integration_mode=args.integration_mode,
        source_uri=args.source_uri,
        license_note=args.license_note,
        compute_sha256=args.compute_sha256,
    )
    output_path = write_wm811k_preflight(report, args.output_path)
    print(f"preflight_report: {output_path}")
    print(f"rows: {report['schema']['row_count']}")
    print(f"classes: {report['labels']['class_count']}")
    print(f"groups: {report['groups']['group_count']}")
    print(f"ready_for_grouped_training: {report['ready_for_grouped_training']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
