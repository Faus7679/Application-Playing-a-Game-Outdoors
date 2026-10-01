#!/usr/bin/env python3
"""Build a categorical Naive Bayes weather-game analysis from CSV or DOCX data."""

import argparse
import csv
import html
import math
import sys
import zipfile
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree


MISSING = "(missing)"
WORD_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def load_rows(path):
    """Read a header row and records from CSV or tables in a DOCX file."""
    path = Path(path)
    if path.suffix.lower() == ".docx":
        try:
            with zipfile.ZipFile(path) as docx:
                document = ElementTree.fromstring(docx.read("word/document.xml"))
        except (OSError, KeyError, zipfile.BadZipFile, ElementTree.ParseError) as error:
            raise ValueError(f"Could not read DOCX document: {error}") from error

        rows = []
        for table in document.iter(f"{WORD_NS}tbl"):
            for row in table.findall(f"{WORD_NS}tr"):
                cells = []
                for cell in row.findall(f"{WORD_NS}tc"):
                    text = "".join(node.text or "" for node in cell.iter(f"{WORD_NS}t"))
                    cells.append(text.strip())
                if any(cells):
                    rows.append(cells)
    else:
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as data_file:
                rows = list(csv.reader(data_file))
        except (OSError, UnicodeError, csv.Error) as error:
            raise ValueError(f"Could not read CSV file: {error}") from error

    rows = [[cell.strip() for cell in row] for row in rows if any(cell.strip() for cell in row)]
    if len(rows) < 2:
        raise ValueError("The input must contain a header row and at least one data row.")

    headers = rows[0]
    if any(not header for header in headers) or len(set(headers)) != len(headers):
        raise ValueError("Column names must be non-empty and unique.")
    if any(len(row) != len(headers) for row in rows[1:]):
        raise ValueError("Every data row must have the same number of columns as the header.")
    return headers, [dict(zip(headers, row)) for row in rows[1:]]


def fit_predict(train_rows, test_row, features, target, classes, categories, alpha=1.0):
    """Predict one class using a categorical Naive Bayes model with Laplace smoothing."""
    counts = Counter(row[target] for row in train_rows)
    total = len(train_rows)
    scores = {}
    for label in classes:
        score = math.log((counts[label] + alpha) / (total + alpha * len(classes)))
        for feature in features:
            value = test_row[feature] or MISSING
            vocabulary_size = len(categories[feature])
            observed = sum(row[target] == label and (row[feature] or MISSING) == value
                          for row in train_rows)
            denominator = counts[label] + alpha * vocabulary_size
            score += math.log((observed + alpha) / denominator)
        scores[label] = score
    return max(classes, key=lambda label: (scores[label], label))


def evaluate(rows, features, target, alpha=1.0):
    """Return leave-one-out predictions and their confusion matrix."""
    classes = sorted({row[target] for row in rows})
    if len(rows) < 2 or len(classes) < 2:
        raise ValueError("Leave-one-out validation requires at least two rows and two outcome classes.")
    categories = {
        feature: {row[feature] or MISSING for row in rows}
        for feature in features
    }
    predictions = []
    for index, row in enumerate(rows):
        training = rows[:index] + rows[index + 1:]
        prediction = fit_predict(training, row, features, target, classes, categories, alpha)
        predictions.append((row[target], prediction))

    matrix = {actual: {predicted: 0 for predicted in classes} for actual in classes}
    for actual, predicted in predictions:
        matrix[actual][predicted] += 1
    correct = sum(actual == predicted for actual, predicted in predictions)
    return classes, categories, predictions, matrix, correct / len(rows)


def write_svg(path, title, body, width=900, height=460):
    content = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">'
        '<style>text{font:14px Arial,sans-serif;fill:#222}.title{font-size:20px;font-weight:bold}'
        '.axis{stroke:#444;stroke-width:1}.grid{stroke:#ddd;stroke-width:1}'
        '.bar{fill:#4778a8}.positive{fill:#5b9b70}.cell{stroke:white;stroke-width:2}</style>'
        f'<text class="title" x="{width / 2:g}" y="30" text-anchor="middle">{html.escape(title)}</text>'
        f'{body}</svg>'
    )
    path.write_text(content, encoding="utf-8")


def plot_class_distribution(path, classes, rows, target):
    counts = Counter(row[target] for row in rows)
    body = ['<line class="axis" x1="90" y1="390" x2="850" y2="390"/>']
    chart_height = 300
    maximum = max(counts.values())
    slot = 740 / len(classes)
    for index, label in enumerate(classes):
        bar_height = chart_height * counts[label] / maximum
        x = 120 + index * slot
        y = 390 - bar_height
        body.extend([
            f'<rect class="bar" x="{x:g}" y="{y:g}" width="{slot * .55:g}" height="{bar_height:g}"/>',
            f'<text x="{x + slot * .275:g}" y="{y - 8:g}" text-anchor="middle">{counts[label]}</text>',
            f'<text x="{x + slot * .275:g}" y="420" text-anchor="middle">{html.escape(label)}</text>',
        ])
    write_svg(path, "Outcome class distribution", "".join(body))


def plot_confusion_matrix(path, classes, matrix):
    cell = min(140, 620 // len(classes))
    left, top = 240, 105
    maximum = max(1, max(matrix[actual][predicted] for actual in classes for predicted in classes))
    body = [f'<text x="{left + (len(classes) * cell) / 2:g}" y="75" text-anchor="middle">Predicted</text>',
            f'<text x="42" y="{top + len(classes) * cell / 2:g}" text-anchor="middle" '
            f'transform="rotate(-90 42 {top + len(classes) * cell / 2:g})">Actual</text>']
    for col, label in enumerate(classes):
        body.append(f'<text x="{left + col * cell + cell / 2:g}" y="{top - 12}" '
                    f'text-anchor="middle">{html.escape(label)}</text>')
    for row_index, actual in enumerate(classes):
        y = top + row_index * cell
        body.append(f'<text x="{left - 12}" y="{y + cell / 2 + 5:g}" text-anchor="end">'
                    f'{html.escape(actual)}</text>')
        for col, predicted in enumerate(classes):
            value = matrix[actual][predicted]
            shade = round(245 - 145 * value / maximum)
            fill = f"rgb({shade},{shade},{min(255, shade + 5)})"
            body.extend([
                f'<rect class="cell" x="{left + col * cell}" y="{y}" width="{cell}" '
                f'height="{cell}" fill="{fill}"/>',
                f'<text x="{left + col * cell + cell / 2:g}" y="{y + cell / 2 + 5:g}" '
                f'text-anchor="middle">{value}</text>',
            ])
    write_svg(path, "Leave-one-out confusion matrix", "".join(body))


def plot_predictor_outcomes(path, rows, features, target, positive_class):
    counts = Counter()
    for row in rows:
        for feature in features:
            value = row[feature] or MISSING
            counts[(feature, value)] += row[target] == positive_class
    ordered = sorted(counts.items(), key=lambda pair: (pair[0][0], pair[0][1]))
    height = max(460, 100 + 28 * len(ordered))
    chart_left, chart_width = 270, 560
    body = []
    maximum = max((count for _, count in ordered), default=0)
    maximum = max(1, maximum)
    for index, ((feature, value), count) in enumerate(ordered):
        y = 75 + index * 28
        label = f"{feature}={value}"
        clipped = label if len(label) <= 34 else label[:31] + "..."
        body.extend([
            f'<text x="{chart_left - 10}" y="{y + 15}" text-anchor="end">{html.escape(clipped)}</text>',
            f'<rect class="positive" x="{chart_left}" y="{y}" width="{chart_width * count / maximum:g}" '
            f'height="18"><title>{html.escape(label)}: {count} positive outcomes</title></rect>',
            f'<text x="{chart_left + chart_width * count / maximum + 6:g}" y="{y + 15}">{count}</text>',
        ])
    write_svg(path, f"Positive outcomes by weather category ({positive_class})",
              "".join(body), height=height)


def generate_report(output, rows, features, target, alpha=1.0):
    classes, categories, predictions, matrix, accuracy = evaluate(rows, features, target, alpha)
    plot_dir = output / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    positive_class = max(classes, key=lambda label: (sum(row[target] == label for row in rows), label))
    plot_class_distribution(plot_dir / "outcome-distribution.svg", classes, rows, target)
    plot_confusion_matrix(plot_dir / "confusion-matrix.svg", classes, matrix)
    plot_predictor_outcomes(plot_dir / "predictor-outcomes.svg", rows, features, target, positive_class)

    result = [
        "# Outdoor Game Weather Classifier",
        "",
        f"Records analyzed: **{len(rows)}**  ",
        f"Outcome column: **{target}**  ",
        f"Predictor columns: **{', '.join(features)}**",
        "",
        "## 1. Naive Bayes model",
        "",
        "A categorical Naive Bayes classifier predicts the game outcome from the weather columns. "
        "For each possible outcome, it combines the smoothed class prior with the smoothed "
        "conditional probabilities of the observed weather categories. Laplace smoothing "
        f"(α = {alpha:g}) avoids zero probabilities for categories not observed with a class. "
        "The prediction is the outcome with the largest log posterior score. The model assumes "
        "weather columns are categorical; numeric values should be converted into meaningful "
        "categories before analysis.",
        "",
        "![Counts of each outcome](plots/outcome-distribution.svg)",
        "",
        "![Positive outcomes for each weather category](plots/predictor-outcomes.svg)",
        "",
        "## 2. Training and validation",
        "",
        f"The model was evaluated with leave-one-out cross-validation (LOOCV): each of the "
        f"{len(rows)} records was held out once, a model was fit using the other "
        f"{len(rows) - 1} records, and a prediction was recorded for the held-out record. "
        "This uses nearly all observations for each training fold and is suitable for small "
        "datasets, though its estimate can have high variance. The reported accuracy is the "
        "fraction of held-out predictions that match their observed outcomes. After validation, "
        "a deployment model should be refit on all available records.",
        "",
        "![LOOCV confusion matrix: rows are actual outcomes and columns are predicted outcomes]"
        "(plots/confusion-matrix.svg)",
        "",
        "## 3. Accuracy",
        "",
        f"LOOCV accuracy = correct held-out predictions / records = "
        f"{sum(a == p for a, p in predictions)} / {len(rows)} = **{accuracy:.1%}**.",
        "",
        "## 4. Assumptions and limitations",
        "",
        "- The recorded examples are representative of the situations in which the classifier will be used.",
        "- Given the outcome, weather predictors are treated as conditionally independent; correlated "
        "conditions (for example, humidity and precipitation) can violate this simplifying assumption.",
        "- Input fields and outcome labels are consistently recorded, and each row is an independent observation.",
        "- Small datasets make validation estimates uncertain. LOOCV can have high variance, and this "
        "evaluation does not assess performance on a separate, future season or location.",
        "- Accuracy weights all mistakes equally and can be misleading for imbalanced outcomes. The "
        "confusion matrix should be considered too; this model does not encode the relative costs of "
        "unsafe conditions versus cancelling a game.",
        "- The model only learns patterns in the supplied historical data. It cannot guarantee safety, "
        "account for unrecorded factors, or establish causal relationships. A human should consider "
        "official alerts and local conditions before deciding to play.",
        "",
        "## References",
        "",
        "Domingos, P., & Pazzani, M. (1997). On the optimality of the simple Bayesian classifier "
        "under zero-one loss. *Machine Learning, 29*, 103–130. "
        "https://doi.org/10.1023/A:1007413511361",
        "",
        "Zhang, H. (2004). The optimality of Naive Bayes. In *Proceedings of the Seventeenth "
        "International Florida Artificial Intelligence Research Society Conference* (pp. 562–567).",
        "",
    ]
    (output / "report.md").write_text("\n".join(result), encoding="utf-8")
    return accuracy


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Train and validate a categorical Naive Bayes game-weather model."
    )
    parser.add_argument("--data", required=True, help="Input CSV or DOCX containing a data table.")
    parser.add_argument("--target", required=True, help="Name of the outcome column (e.g. Play).")
    parser.add_argument("--output", default="weather-analysis", help="Directory for report and plots.")
    parser.add_argument("--alpha", type=float, default=1.0, help="Laplace smoothing parameter (positive).")
    args = parser.parse_args(argv)
    if not math.isfinite(args.alpha) or args.alpha <= 0:
        parser.error("--alpha must be a positive finite number.")

    try:
        headers, rows = load_rows(args.data)
        if args.target not in headers:
            raise ValueError(f"Outcome column {args.target!r} is not in the data header.")
        if any(not row[args.target] for row in rows):
            raise ValueError("Outcome values cannot be blank.")
        features = [header for header in headers if header != args.target]
        if not features:
            raise ValueError("The input must contain at least one weather predictor column.")
        accuracy = generate_report(Path(args.output), rows, features, args.target, args.alpha)
    except (OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2

    print(f"Analyzed {len(rows)} records; LOOCV accuracy: {accuracy:.1%}.")
    print(f"Report and plots written to {Path(args.output).resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
