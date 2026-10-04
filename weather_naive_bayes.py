"""Categorical Naive Bayes on the supplied 14-row CST-570 weather dataset.

Standard library only. Plots are written as SVG files to ``plots/``.
"""
import math
import sys
from collections import Counter
from pathlib import Path
from xml.sax.saxutils import escape

FEATURES = ("weather", "temperature", "humidity", "windy")
TARGET = "play"
SCHEMA = {
    "weather": ("sunny", "overcast", "rainy"),
    "temperature": ("above average", "average", "below average"),
    "humidity": ("low", "high"),
    "windy": ("calm", "windy"),
}
CLASSES = ("yes", "no")
POSITIVE = "yes"

# (weather, temperature, humidity, windy, play)
ROWS = [
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

PLOT_FILES = (
    "class_priors_and_smoothing.svg",
    "loocv_predictions.svg",
    "confusion_and_baseline.svg",
    "feature_outcome_distribution.svg",
)


def validate_rows(rows):
    for r in rows:
        if len(r) != len(FEATURES) + 1:
            raise ValueError("row has wrong length: %r" % (r,))
        for f, v in zip(FEATURES, r):
            if v not in SCHEMA[f]:
                raise ValueError("value %r not in declared domain of %s" % (v, f))
        if r[-1] not in CLASSES:
            raise ValueError("unknown class %r" % (r[-1],))


class NaiveBayes:
    """Categorical Naive Bayes with add-one smoothing over the declared schema."""

    def __init__(self, schema=SCHEMA, classes=CLASSES, alpha=1):
        self.schema, self.classes, self.alpha = schema, classes, alpha

    def fit(self, rows):
        validate_rows(rows)
        self.n = len(rows)
        self.class_counts = Counter(r[-1] for r in rows)
        self.counts = Counter()
        for r in rows:
            for f, v in zip(FEATURES, r):
                self.counts[(f, v, r[-1])] += 1
        return self

    def prior(self, c):
        return self.class_counts[c] / self.n

    def cond(self, f, v, c):
        k = len(self.schema[f])
        return (self.counts[(f, v, c)] + self.alpha) / (self.class_counts[c] + self.alpha * k)

    def log_scores(self, x):
        out = {}
        for c in self.classes:
            s = math.log(self.prior(c)) if self.class_counts[c] else float("-inf")
            for f, v in zip(FEATURES, x):
                s += math.log(self.cond(f, v, c))
            out[c] = s
        return out

    def predict(self, x):
        scores = self.log_scores(x)
        return max(self.classes, key=lambda c: scores[c])  # ties -> first declared class


def loocv(rows):
    """Hold out each row once; fit on the other rows with the declared schema."""
    results = []
    for i, row in enumerate(rows):
        model = NaiveBayes().fit(rows[:i] + rows[i + 1:])
        pred = model.predict(row[:-1])
        results.append((i + 1, row, pred, model.log_scores(row[:-1])))
    return results


def metrics(rows, results):
    tp = sum(1 for _, r, p, _s in results if r[-1] == POSITIVE and p == POSITIVE)
    fn = sum(1 for _, r, p, _s in results if r[-1] == POSITIVE and p != POSITIVE)
    fp = sum(1 for _, r, p, _s in results if r[-1] != POSITIVE and p == POSITIVE)
    tn = sum(1 for _, r, p, _s in results if r[-1] != POSITIVE and p != POSITIVE)
    n = len(results)
    counts = Counter(r[-1] for r in rows)
    majority = max(CLASSES, key=lambda c: counts[c])
    return {
        "n": n, "correct": tp + tn, "accuracy": (tp + tn) / n,
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "majority_class": majority, "baseline_accuracy": counts[majority] / n,
        "class_counts": dict(counts),
    }


# ---------------------------------------------------------------- SVG helpers
def _svg(width, height, body):
    return ('<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" '
            'viewBox="0 0 %d %d" font-family="sans-serif" font-size="12">'
            '<rect width="100%%" height="100%%" fill="white"/>%s</svg>\n'
            % (width, height, width, height, body))


def _text(x, y, s, size=12, anchor="start", weight="normal", fill="black"):
    return ('<text x="%.1f" y="%.1f" font-size="%d" text-anchor="%s" '
            'font-weight="%s" fill="%s">%s</text>'
            % (x, y, size, anchor, weight, fill, escape(str(s))))


def _rect(x, y, w, h, fill, stroke="none"):
    return ('<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" fill="%s" stroke="%s"/>'
            % (x, y, w, h, fill, stroke))


def _bars(x0, y0, w, h, labels, values, colors, maxv, fmt):
    """Vertical bar chart in a w x h box whose top-left is (x0, y0)."""
    body = ['<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="black"/>'
            % (x0, y0 + h, x0 + w, y0 + h)]
    slot = w / len(values)
    for i, (lab, v, col) in enumerate(zip(labels, values, colors)):
        bh = h * v / maxv if maxv else 0
        bx = x0 + i * slot + slot * 0.2
        body.append(_rect(bx, y0 + h - bh, slot * 0.6, bh, col))
        body.append(_text(bx + slot * 0.3, y0 + h - bh - 4, fmt(v), anchor="middle"))
        body.append(_text(bx + slot * 0.3, y0 + h + 16, lab, anchor="middle"))
    return "".join(body)


YES_C, NO_C = "#4c9f70", "#d9695f"


def plot_priors(rows, outdir):
    m = NaiveBayes().fit(rows)
    counts = [m.class_counts[c] for c in CLASSES]
    n_yes, n_sun = m.class_counts["yes"], m.counts[("weather", "sunny", "yes")]
    k = len(SCHEMA["weather"])
    raw, smooth = n_sun / n_yes, m.cond("outlook", "sunny", "yes")
    body = [_text(300, 24, "Class counts and priors (CST-570 dataset, n=%d)" % m.n, 14, "middle", "bold")]
    body.append(_bars(60, 50, 220, 180, list(CLASSES), counts, [YES_C, NO_C], max(counts), str))
    for i, c in enumerate(CLASSES):
        body.append(_text(60 + 110 * i + 55, 262, "prior %d/%d = %.3f" % (counts[i], m.n, counts[i] / m.n), anchor="middle"))
    body.append(_bars(340, 50, 220, 180, ["raw %d/%d" % (n_sun, n_yes), "smoothed"],
                      [raw, smooth], ["#999999", "#4c78a8"], 0.5, lambda v: "%.3f" % v))
    body.append(_text(450, 262, "P(weather=sunny|yes) = (%d+1)/(%d+%d) = %.4f" % (n_sun, n_yes, k, smooth), anchor="middle"))
    (outdir / PLOT_FILES[0]).write_text(_svg(620, 300, "".join(body)), encoding="utf-8")


def plot_loocv(results, outdir):
    row_h, top = 24, 70
    h = top + row_h * len(results) + 30
    body = [_text(350, 24, "CST-570 LOOCV: each row held out once, fit on the other 13", 14, "middle", "bold")]
    for j, hd in enumerate(("fold", "held-out row", "actual", "predicted", "result")):
        body.append(_text((30, 90, 400, 470, 560)[j], top - 6, hd, weight="bold"))
    for i, (fold, row, pred, _s) in enumerate(results):
        y = top + i * row_h
        ok = pred == row[-1]
        body.append(_rect(20, y, 660, row_h - 2, "#e3f2e8" if ok else "#fbe3e0"))
        body.append(_text(30, y + 16, fold))
        body.append(_text(90, y + 16, ", ".join(row[:-1])))
        body.append(_text(400, y + 16, row[-1]))
        body.append(_text(470, y + 16, pred))
        body.append(_text(560, y + 16, "correct" if ok else "wrong"))
    (outdir / PLOT_FILES[1]).write_text(_svg(700, h, "".join(body)), encoding="utf-8")


def plot_confusion(met, outdir):
    body = [_text(350, 24, "LOOCV confusion matrix and baseline (yes = positive)", 14, "middle", "bold")]
    cells = [(met["tp"], "TP", 0, 0), (met["fn"], "FN", 1, 0), (met["fp"], "FP", 0, 1), (met["tn"], "TN", 1, 1)]
    body.append(_text(140, 56, "predicted yes", anchor="middle"))
    body.append(_text(240, 56, "predicted no", anchor="middle"))
    body.append(_text(30, 105, "actual yes"))
    body.append(_text(30, 205, "actual no"))
    for v, name, cx, cy in cells:
        good = (name in ("TP", "TN"))
        body.append(_rect(90 + 100 * cx, 70 + 100 * cy, 100, 100, "#cfe8d8" if good else "#f6d3cf", "black"))
        body.append(_text(140 + 100 * cx, 125 + 100 * cy, v, 24, "middle", "bold"))
        body.append(_text(140 + 100 * cx, 148 + 100 * cy, name, anchor="middle"))
    body.append(_bars(380, 70, 260, 200, ["LOOCV model", "majority (%s)" % met["majority_class"]],
                      [met["accuracy"], met["baseline_accuracy"]], ["#4c78a8", "#999999"], 1.0,
                      lambda v: "%.1f%%" % (100 * v)))
    (outdir / PLOT_FILES[2]).write_text(_svg(700, 345, "".join(body)), encoding="utf-8")


def plot_features(rows, outdir):
    cats = [(f, v) for f in FEATURES for v in SCHEMA[f]]
    row_h, top, scale = 22, 60, 20
    h = top + row_h * len(cats) + 40
    body = [_text(300, 24, "Outcome distribution by feature category", 14, "middle", "bold"),
            _rect(420, 34, 12, 12, YES_C), _text(436, 45, "yes"),
            _rect(480, 34, 12, 12, NO_C), _text(496, 45, "no")]
    for i, (f, v) in enumerate(cats):
        y = top + i * row_h
        ny = sum(1 for r in rows if r[FEATURES.index(f)] == v and r[-1] == "yes")
        nn = sum(1 for r in rows if r[FEATURES.index(f)] == v and r[-1] == "no")
        body.append(_text(20, y + 14, "%s = %s" % (f, v)))
        body.append(_rect(160, y, ny * scale, row_h - 4, YES_C))
        body.append(_rect(160 + ny * scale, y, nn * scale, row_h - 4, NO_C))
        body.append(_text(165 + (ny + nn) * scale, y + 14, "%d yes / %d no" % (ny, nn)))
    (outdir / PLOT_FILES[3]).write_text(_svg(600, h, "".join(body)), encoding="utf-8")


def generate_plots(rows, results, met, outdir):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    plot_priors(rows, outdir)
    plot_loocv(results, outdir)
    plot_confusion(met, outdir)
    plot_features(rows, outdir)
    return [outdir / f for f in PLOT_FILES]


def main(argv=None):
    outdir = Path(__file__).resolve().parent / "plots"
    rows = list(ROWS)
    results = loocv(rows)
    met = metrics(rows, results)
    print("Class counts:", met["class_counts"])
    print("Fold  held-out row                          actual  predicted  log-scores")
    for fold, row, pred, sc in results:
        print("%4d  %-36s  %-6s  %-9s  yes=%.4f no=%.4f"
              % (fold, ", ".join(row[:-1]), row[-1], pred, sc["yes"], sc["no"]))
    print("LOOCV accuracy: %d/%d = %.1f%%" % (met["correct"], met["n"], 100 * met["accuracy"]))
    print("Confusion (yes positive): TP=%d FN=%d FP=%d TN=%d" % (met["tp"], met["fn"], met["fp"], met["tn"]))
    print("Majority baseline (%s): %d/%d = %.1f%%" % (
        met["majority_class"], round(met["baseline_accuracy"] * met["n"]), met["n"], 100 * met["baseline_accuracy"]))
    full = NaiveBayes().fit(rows)
    print("P(weather=sunny|yes) = (%d+1)/(%d+%d) = %.4f" % (
        full.counts[("weather", "sunny", "yes")], full.class_counts["yes"], len(SCHEMA["weather"]),
        full.cond("outlook", "sunny", "yes")))
    for p in generate_plots(rows, results, met, outdir):
        print("Wrote", p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
