"""Generate a Markdown summary report from pipeline CSV outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.reporting import build_summary_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a Markdown summary report.")
    parser.add_argument("--reports-dir", type=Path, default=Path("outputs/reports"))
    parser.add_argument("--monitoring-dir", type=Path, default=Path("outputs/reports/monitoring"))
    parser.add_argument("--output-path", type=Path, default=Path("outputs/reports/summary_report.md"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    build_summary_report(
        reports_dir=args.reports_dir,
        monitoring_dir=args.monitoring_dir,
        output_path=args.output_path,
    )
    print(f"Summary report written to: {args.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
