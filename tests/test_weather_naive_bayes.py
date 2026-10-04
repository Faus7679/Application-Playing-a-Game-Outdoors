import tempfile
import unittest
from collections import Counter
from fractions import Fraction
from pathlib import Path

import weather_naive_bayes as wnb


def independent_loocv_accuracy(rows):
    """Exact-fraction recomputation, independent of the module's model class."""
    correct = 0
    for i, row in enumerate(rows):
        train = rows[:i] + rows[i + 1:]
        best = None
        for c in ("yes", "no"):
            nc = sum(1 for r in train if r[-1] == c)
            p = Fraction(nc, len(train))
            for j, f in enumerate(wnb.FEATURES):
                n = sum(1 for r in train if r[-1] == c and r[j] == row[j])
                p *= Fraction(n + 1, nc + len(wnb.SCHEMA[f]))
            if best is None or p > best[0]:
                best = (p, c)
        correct += best[1] == row[-1]
    return correct


class WeatherNaiveBayesTests(unittest.TestCase):
    def test_dataset_shape_and_counts(self):
        self.assertEqual(len(wnb.ROWS), 14)
        wnb.validate_rows(wnb.ROWS)
        self.assertEqual(Counter(r[-1] for r in wnb.ROWS), {"yes": 9, "no": 5})

    def test_smoothing_example(self):
        m = wnb.NaiveBayes().fit(wnb.ROWS)
        self.assertEqual(m.counts[("outlook", "sunny", "yes")], 2)
        self.assertAlmostEqual(m.cond("outlook", "sunny", "yes"), 0.25)

    def test_unseen_value_not_zero(self):
        m = wnb.NaiveBayes().fit([r for r in wnb.ROWS if r[0] != "overcast"])
        self.assertGreater(m.cond("outlook", "overcast", "no"), 0)

    def test_loocv_and_metrics(self):
        res = wnb.loocv(wnb.ROWS)
        self.assertEqual(len(res), 14)
        met = wnb.metrics(wnb.ROWS, res)
        self.assertEqual(met["correct"], independent_loocv_accuracy(wnb.ROWS))
        self.assertEqual((met["tp"], met["fn"], met["fp"], met["tn"]), (6, 3, 4, 1))
        self.assertEqual(met["correct"], 7)
        self.assertAlmostEqual(met["accuracy"], 0.5)
        self.assertAlmostEqual(met["baseline_accuracy"], 9 / 14)

    def test_plots_generated(self):
        res = wnb.loocv(wnb.ROWS)
        met = wnb.metrics(wnb.ROWS, res)
        with tempfile.TemporaryDirectory() as d:
            paths = wnb.generate_plots(wnb.ROWS, res, met, d)
            self.assertEqual({p.name for p in paths}, set(wnb.PLOT_FILES))
            for p in paths:
                text = Path(p).read_text(encoding="utf-8")
                self.assertTrue(text.startswith("<svg"))
                self.assertIn("not the course dataset", text)


if __name__ == "__main__":
    unittest.main()
