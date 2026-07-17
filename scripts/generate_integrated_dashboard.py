"""Generate a standalone dashboard from available analysis outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.integrated_dashboard import build_integrated_dashboard


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the integrated manufacturing AI dashboard.")
    parser.add_argument("--reports-dir", type=Path, default=Path("outputs/reports"))
    parser.add_argument("--wafer-dir", type=Path, default=Path("outputs/wafer_maps"))
    parser.add_argument("--equipment-dir", type=Path, default=Path("outputs/equipment_anomalies"))
    parser.add_argument("--output-path", type=Path, default=Path("outputs/integrated_dashboard/index.html"))
    parser.add_argument("--secom-mode", choices=["unknown", "synthetic", "demo", "real"], default="unknown")
    parser.add_argument("--wafer-mode", choices=["unknown", "synthetic", "demo", "real"], default="unknown")
    parser.add_argument("--equipment-mode", choices=["unknown", "synthetic", "demo", "real"], default="unknown")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    build_integrated_dashboard(
        reports_dir=args.reports_dir,
        wafer_dir=args.wafer_dir,
        equipment_dir=args.equipment_dir,
        output_path=args.output_path,
        secom_mode=args.secom_mode,
        wafer_mode=args.wafer_mode,
        equipment_mode=args.equipment_mode,
    )
    print(f"Integrated dashboard written to: {args.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
