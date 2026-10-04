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
    def test_supplied_dataset_and_schema(self):
        expected_rows = [
            ("sunny", "above average", "low", "calm", "no"),
            ("sunny", "above average", "low", "windy", "no"),
            ("overcast", "above average", "low", "calm", "yes"),
            ("rainy", "average", "low", "calm", "yes"),
            ("rainy", "average", "high", "calm", "yes"),
            ("rainy", "below average", "high", "windy", "no"),
            ("overcast", "below average", "high", "windy", "yes"),
            ("sunny", "below average", "low", "calm", "no"),
            ("sunny", "average", "high", "calm", "yes"),
            ("rainy", "below average", "high", "calm", "yes"),
            ("sunny", "average", "low", "windy", "yes"),
            ("overcast", "average", "low", "windy", "yes"),
            ("overcast", "above average", "high", "calm", "yes"),
            ("rainy", "average", "low", "windy", "no"),
        ]
        self.assertEqual(wnb.FEATURES, ("weather", "temperature", "humidity", "windy"))
        self.assertEqual(wnb.SCHEMA, {
            "weather": ("sunny", "overcast", "rainy"),
            "temperature": ("above average", "average", "below average"),
            "humidity": ("low", "high"),
            "windy": ("calm", "windy"),
        })
        self.assertEqual(wnb.ROWS, expected_rows)
        wnb.validate_rows(wnb.ROWS)
        self.assertEqual(Counter(r[-1] for r in wnb.ROWS), {"yes": 9, "no": 5})

    def test_smoothing_example(self):
        m = wnb.NaiveBayes().fit(wnb.ROWS)
        self.assertAlmostEqual(m.prior("yes"), 9 / 14)
        self.assertAlmostEqual(m.prior("no"), 5 / 14)
        self.assertEqual(m.counts[("weather", "sunny", "yes")], 2)
        self.assertAlmostEqual(m.cond("weather", "sunny", "yes"), 0.25)

    def test_unseen_value_not_zero(self):
        m = wnb.NaiveBayes().fit([r for r in wnb.ROWS if r[0] != "overcast"])
        self.assertGreater(m.cond("weather", "overcast", "no"), 0)

    def test_loocv_and_metrics(self):
        res = wnb.loocv(wnb.ROWS)
        self.assertEqual(len(res), 14)
        self.assertEqual([prediction for _, _, prediction, _ in res],
                         ["yes", "no", "yes", "yes", "yes", "yes", "yes",
                          "yes", "yes", "yes", "no", "yes", "yes", "yes"])
        met = wnb.metrics(wnb.ROWS, res)
        self.assertEqual(met["correct"], independent_loocv_accuracy(wnb.ROWS))
        self.assertEqual((met["tp"], met["fn"], met["fp"], met["tn"]), (8, 1, 4, 1))
        self.assertEqual(met["correct"], 9)
        self.assertAlmostEqual(met["accuracy"], 9 / 14)
        self.assertEqual(met["majority_class"], "yes")
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
                self.assertNotIn("benchmark", text.lower())
                self.assertNotIn("not the course dataset", text.lower())
            self.assertIn("CST-570 dataset", Path(paths[0]).read_text(encoding="utf-8"))
            self.assertIn("weather=sunny", Path(paths[0]).read_text(encoding="utf-8"))
            self.assertIn(">8<", Path(paths[2]).read_text(encoding="utf-8"))

    def test_report_documents_supplied_data_results(self):
        report = Path(wnb.__file__).resolve().parent.joinpath("report.md").read_text(encoding="utf-8")
        self.assertIn("9/14", report)
        self.assertIn("64.3%", report)
        self.assertIn("TP = 8, FN = 1", report)
        self.assertIn("FP = 4, and TN = 1", report)
        self.assertIn("majority-class baseline", report)
        self.assertIn("retrospective label prediction", report.lower())
        self.assertNotIn("Play Tennis", report)
        self.assertNotIn("not the course dataset", report)


if __name__ == "__main__":
    unittest.main()
