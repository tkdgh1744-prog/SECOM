from __future__ import annotations

import unittest

import pandas as pd

from src.secom_training import parse_thresholds, rank_results, select_top_result


class SecomTrainingTests(unittest.TestCase):
    def test_parse_thresholds_accepts_csv_and_iterable(self) -> None:
        self.assertEqual(parse_thresholds("0.2, 0.5,0.8"), [0.2, 0.5, 0.8])
        self.assertEqual(parse_thresholds([0.1, 0.9]), [0.1, 0.9])

    def test_parse_thresholds_rejects_invalid_values(self) -> None:
        with self.assertRaises(ValueError):
            parse_thresholds("0.1,1.2")

    def test_rank_results_sorts_by_priority_metrics(self) -> None:
        results = pd.DataFrame(
            [
                {"model": "A", "fail_recall": 0.8, "fail_f1": 0.2, "pr_auc": 0.4},
                {"model": "B", "fail_recall": 0.8, "fail_f1": 0.5, "pr_auc": 0.3},
                {"model": "C", "fail_recall": 0.6, "fail_f1": 0.9, "pr_auc": 0.9},
            ]
        )

        ranked = rank_results(results)

        self.assertEqual(ranked["model"].tolist(), ["B", "A", "C"])

    def test_select_top_result_returns_best_row(self) -> None:
        results = pd.DataFrame(
            [
                {"model": "baseline", "fail_recall": 0.0, "fail_f1": 0.0, "pr_auc": 0.1},
                {"model": "balanced", "fail_recall": 0.7, "fail_f1": 0.4, "pr_auc": 0.3},
            ]
        )

        selected = select_top_result(results)

        self.assertEqual(selected.name, "balanced")
        self.assertEqual(selected.ranking_columns, ("fail_recall", "fail_f1", "pr_auc"))


if __name__ == "__main__":
    unittest.main()
