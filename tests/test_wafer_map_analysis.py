from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.analyze_wafer_maps import (
    load_records_from_args,
    make_demo_records,
    parse_non_negative_int,
    parse_positive_int,
)
from src.wafer_map_analysis import (
    WaferMapRecord,
    connected_component_sizes,
    coordinate_table_to_wafer_records,
    extract_wafer_map_feature_table,
    infer_wafer_masks,
    run_wafer_map_analysis,
    wafer_similarity_pairs,
)


class WaferMapAnalysisTests(unittest.TestCase):
    def test_record_limit_parsers_define_zero_and_negative_behavior(self) -> None:
        self.assertEqual(parse_positive_int("3"), 3)
        self.assertEqual(parse_non_negative_int("0"), 0)

        for value in ("0", "-1"):
            with self.subTest(parser="positive", value=value):
                with self.assertRaises(argparse.ArgumentTypeError):
                    parse_positive_int(value)
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_non_negative_int("-1")

    def test_demo_record_limit_is_applied(self) -> None:
        args = argparse.Namespace(demo=True, max_records=2, input_path=None)

        records = load_records_from_args(args)

        self.assertEqual([record.wafer_id for record in records], ["demo_center", "demo_edge_ring"])

    def test_infer_wm811k_masks_uses_two_as_defect(self) -> None:
        wafer_map = np.array(
            [
                [0, 1, 0],
                [1, 2, 1],
                [0, 1, 0],
            ]
        )

        valid_mask, defect_mask = infer_wafer_masks(wafer_map)

        self.assertEqual(int(valid_mask.sum()), 5)
        self.assertEqual(int(defect_mask.sum()), 1)

    def test_connected_component_sizes_uses_eight_neighbors(self) -> None:
        mask = np.array(
            [
                [1, 0, 0, 1],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
            ],
            dtype=bool,
        )

        self.assertEqual(sorted(connected_component_sizes(mask)), [1, 1, 2])

    def test_extract_features_assigns_expected_demo_patterns(self) -> None:
        records = make_demo_records()
        features = extract_wafer_map_feature_table(records)
        pattern_by_id = dict(zip(features["wafer_id"], features["heuristic_pattern"]))

        self.assertEqual(pattern_by_id["demo_center"], "Center")
        self.assertEqual(pattern_by_id["demo_edge_ring"], "Edge-Ring")
        self.assertEqual(pattern_by_id["demo_edge_local"], "Edge-Local")
        self.assertEqual(pattern_by_id["demo_scratch"], "Scratch")
        self.assertEqual(pattern_by_id["demo_near_full"], "Near-Full")
        self.assertIn("direction_bias", features.columns)
        self.assertIn("cluster_count", features.columns)

    def test_coordinate_table_to_records_builds_maps(self) -> None:
        df = pd.DataFrame(
            {
                "wafer_id": ["W1", "W1", "W1", "W2"],
                "x": [0, 1, 2, 0],
                "y": [0, 0, 0, 0],
                "is_defect": [False, True, False, True],
                "pattern": ["Local", "Local", "Local", "Center"],
            }
        )

        records = coordinate_table_to_wafer_records(
            df,
            defect_col="is_defect",
            label_col="pattern",
        )

        self.assertEqual(len(records), 2)
        self.assertEqual(records[0].label, "Local")
        self.assertEqual(int((records[0].wafer_map == 2).sum()), 1)

    def test_similarity_pairs_finds_nearest_maps(self) -> None:
        map_a = np.zeros((5, 5), dtype=int)
        map_b = np.zeros((5, 5), dtype=int)
        map_c = np.zeros((5, 5), dtype=int)
        map_a[:, :] = 1
        map_b[:, :] = 1
        map_c[:, :] = 1
        map_a[2, 2] = 2
        map_b[2, 2] = 2
        map_c[0, 0] = 2
        pairs = wafer_similarity_pairs(
            [
                WaferMapRecord("A", map_a),
                WaferMapRecord("B", map_b),
                WaferMapRecord("C", map_c),
            ],
            top_k=1,
        )

        nearest_to_a = pairs.loc[pairs["wafer_id"] == "A"].iloc[0]
        self.assertEqual(nearest_to_a["similar_wafer_id"], "B")
        self.assertGreater(nearest_to_a["similarity"], 0.9)

    def test_run_wafer_map_analysis_writes_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            outputs = run_wafer_map_analysis(
                make_demo_records(),
                output_dir=Path(tmpdir),
                n_clusters=3,
                max_images=2,
            )

            self.assertTrue(outputs["features"].exists())
            self.assertTrue(outputs["pattern_summary"].exists())
            self.assertTrue(outputs["report"].exists())
            features = pd.read_csv(outputs["features"])
            self.assertIn("pattern_cluster", features.columns)

    def test_similarity_limit_zero_skips_pairwise_search(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            outputs = run_wafer_map_analysis(
                make_demo_records(),
                output_dir=Path(tmpdir),
                n_clusters=3,
                max_images=0,
                similarity_max_records=0,
            )

            pairs = pd.read_csv(outputs["similarity_pairs"])
            self.assertTrue(pairs.empty)
            self.assertEqual(
                list(pairs.columns),
                ["wafer_id", "similar_wafer_id", "similarity", "rank"],
            )

    def test_negative_similarity_limit_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(ValueError, "similarity_max_records"):
                run_wafer_map_analysis(
                    make_demo_records(),
                    output_dir=Path(tmpdir),
                    similarity_max_records=-1,
                )


if __name__ == "__main__":
    unittest.main()
