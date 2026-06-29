from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from pathbsr.data import load_dataset_with_audit, read_triplets, remove_train_overlap
from pathbsr.evaluation import filtered_rank


class DataAuditTests(unittest.TestCase):
    def test_remove_train_overlap_only_changes_evaluation_queries(self):
        train = [("h", "r", "t"), ("x", "r", "y")]
        evaluation = [("h", "r", "t"), ("z", "r", "u")]

        cleaned, removed = remove_train_overlap(train, evaluation)

        self.assertEqual(cleaned, [("z", "r", "u")])
        self.assertEqual(removed, 1)
        self.assertEqual(train, [("h", "r", "t"), ("x", "r", "y")])

    def test_read_triplets_fails_on_malformed_row_with_file_and_line(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bad.txt"
            path.write_text("h\tt\tr\nbroken-row\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, r"bad\.txt:2: expected 3 tab-separated fields"):
                read_triplets(path)

    def test_load_dataset_with_audit_reports_duplicates_and_overlap(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "datasets" / "Toy"
            root.mkdir(parents=True)
            (root / "train.txt").write_text("h\tt\tr\nh\tt\tr\nu\tv\ts\n", encoding="utf-8")
            (root / "valid.txt").write_text("h\tt\tr\nx\ty\tp\n", encoding="utf-8")
            (root / "test.txt").write_text("x\ty\tp\nm\tn\tq\n", encoding="utf-8")

            train, valid, test, audit = load_dataset_with_audit(root.parent, "Toy")

            self.assertEqual(len(train), 3)
            self.assertEqual(len(valid), 2)
            self.assertEqual(len(test), 2)
            self.assertEqual(audit["splits"]["train"]["duplicate_count"], 1)
            self.assertEqual(audit["splits"]["train"]["unique_count"], 2)
            self.assertEqual(audit["overlap"]["train_valid"]["count"], 1)
            self.assertEqual(audit["overlap"]["valid_test"]["count"], 1)
            self.assertEqual(audit["overlap"]["train_test"]["count"], 0)


class EvaluationGuardrailTests(unittest.TestCase):
    def test_filtered_rank_raises_on_missing_target(self):
        scores = np.array([0.1, 0.2], dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "missing from the ranking vocabulary"):
            filtered_rank(scores, ("h", "r"), "c", {"a": 0, "b": 1}, {})

    def test_filtered_rank_raises_on_invalid_tie_mode(self):
        scores = np.array([0.1, 0.2], dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "Unsupported tie_mode"):
            filtered_rank(scores, ("h", "r"), "a", {"a": 0, "b": 1}, {}, tie_mode="median")


if __name__ == "__main__":
    unittest.main()
