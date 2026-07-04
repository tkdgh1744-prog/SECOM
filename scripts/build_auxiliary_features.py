"""Build wafer and equipment feature CSV files from raw auxiliary data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.data_contracts import validate_equipment_events_contract, validate_wafer_inspection_contract
from src.equipment_features import equipment_event_features
from src.wafer_features import heuristic_wafer_pattern_label, wafer_defect_features


def parse_optional_path(value: str | None) -> Path | None:
    """Parse optional path arguments."""
    if value is None or not str(value).strip():
        return None
    return Path(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build wafer and equipment feature CSV files.")
    parser.add_argument("--wafer-input", type=parse_optional_path, default=None)
    parser.add_argument("--equipment-input", type=parse_optional_path, default=None)
    parser.add_argument(
        "--wafer-output",
        type=Path,
        default=Path("data/raw/wafer_features.csv"),
        help="Output CSV path for wafer-level features.",
    )
    parser.add_argument(
        "--equipment-output",
        type=Path,
        default=Path("data/raw/equipment_features.csv"),
        help="Output CSV path for equipment-level features.",
    )
    parser.add_argument(
        "--add-wafer-pattern-label",
        action="store_true",
        help="Add heuristic wafer pattern labels to wafer feature output.",
    )
    return parser.parse_args()


def build_wafer_features_file(
    wafer_input: Path,
    wafer_output: Path,
    add_pattern_label: bool = False,
) -> Path:
    """Build a wafer-level feature CSV from coordinate-level inspection data."""
    if not wafer_input.exists():
        raise FileNotFoundError(f"Wafer input not found: {wafer_input}")

    wafer_df = pd.read_csv(wafer_input)
    validate_wafer_inspection_contract(wafer_df).raise_if_failed()
    features = wafer_defect_features(wafer_df)
    if add_pattern_label:
        features["pattern_label"] = heuristic_wafer_pattern_label(features)

    wafer_output.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(wafer_output, index=False)
    return wafer_output


def build_equipment_features_file(equipment_input: Path, equipment_output: Path) -> Path:
    """Build an equipment-level feature CSV from event log data."""
    if not equipment_input.exists():
        raise FileNotFoundError(f"Equipment input not found: {equipment_input}")

    equipment_df = pd.read_csv(equipment_input)
    validate_equipment_events_contract(equipment_df).raise_if_failed()
    features = equipment_event_features(equipment_df)

    equipment_output.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(equipment_output, index=False)
    return equipment_output


def build_auxiliary_features(
    wafer_input: Path | None = None,
    equipment_input: Path | None = None,
    wafer_output: Path = Path("data/raw/wafer_features.csv"),
    equipment_output: Path = Path("data/raw/equipment_features.csv"),
    add_wafer_pattern_label: bool = False,
) -> dict[str, Path]:
    """Build available auxiliary feature files and return output paths."""
    outputs: dict[str, Path] = {}

    if wafer_input is not None:
        outputs["wafer"] = build_wafer_features_file(
            wafer_input=wafer_input,
            wafer_output=wafer_output,
            add_pattern_label=add_wafer_pattern_label,
        )

    if equipment_input is not None:
        outputs["equipment"] = build_equipment_features_file(
            equipment_input=equipment_input,
            equipment_output=equipment_output,
        )

    if not outputs:
        raise ValueError("At least one of --wafer-input or --equipment-input is required.")

    return outputs


def main() -> int:
    args = parse_args()
    outputs = build_auxiliary_features(
        wafer_input=args.wafer_input,
        equipment_input=args.equipment_input,
        wafer_output=args.wafer_output,
        equipment_output=args.equipment_output,
        add_wafer_pattern_label=args.add_wafer_pattern_label,
    )
    for name, path in outputs.items():
        print(f"{name} features written to: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
