from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from training_evaluation import (
    evaluate_confusion_matrix,
    load_training_evaluation,
    save_evaluation_artifacts,
)


class TrainingEvaluationTests(unittest.TestCase):
    def test_binary_metrics_and_prevalence_bias(self):
        matrix = np.array([[8, 2], [1, 9]], dtype=int)
        result = evaluate_confusion_matrix(matrix, ["A", "B"])

        self.assertEqual(result["n_samples"], 20)
        self.assertAlmostEqual(result["accuracy"], 0.85)
        self.assertAlmostEqual(result["per_class"][0]["recall"], 0.8)
        self.assertAlmostEqual(result["per_class"][0]["precision"], 8 / 9)
        self.assertAlmostEqual(result["per_class"][1]["recall"], 0.9)
        self.assertAlmostEqual(result["per_class"][1]["precision"], 9 / 11)
        self.assertAlmostEqual(result["per_class"][0]["bias_pp"], -5.0)
        self.assertAlmostEqual(result["per_class"][1]["bias_pp"], 5.0)
        self.assertAlmostEqual(result["mean_abs_bias_pp"], 5.0)
        self.assertAlmostEqual(
            result["macro_f1"],
            (2 * (8 / 9) * 0.8 / ((8 / 9) + 0.8) + 2 * (9 / 11) * 0.9 / ((9 / 11) + 0.9)) / 2,
        )

    def test_zero_denominators_remain_finite(self):
        matrix = np.array([[5, 0], [0, 0]], dtype=int)
        result = evaluate_confusion_matrix(matrix, ["A", "B"])
        for row in result["per_class"]:
            for key in ("precision", "recall", "f1", "bias_pp"):
                self.assertTrue(np.isfinite(row[key]))

    def test_artifacts_roundtrip_uses_label_index_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            matrix = np.array([[3, 1], [2, 4]], dtype=int)
            labels = {"Taxon_B": 1, "Taxon_A": 0}
            # train.py already writes matrix/labels; reproduce those canonical files.
            np.savetxt(root / "matrix_demo.csv", matrix, delimiter=", ")
            (root / "labels_demo.json").write_text(json.dumps(labels), encoding="utf-8")

            paths = save_evaluation_artifacts(
                matrix, labels, root, "demo", test_loss=0.42, configured_macro_divisor=-1
            )
            self.assertTrue(paths["summary"].is_file())
            self.assertTrue(paths["metrics"].is_file())

            loaded_matrix, names, result = load_training_evaluation(root, "demo")
            np.testing.assert_array_equal(loaded_matrix, matrix)
            self.assertEqual(names, ["Taxon_A", "Taxon_B"])
            self.assertAlmostEqual(result["test_loss"], 0.42)

    def test_shape_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            evaluate_confusion_matrix([[1, 2], [3, 4]], ["only_one"])


if __name__ == "__main__":
    unittest.main()
