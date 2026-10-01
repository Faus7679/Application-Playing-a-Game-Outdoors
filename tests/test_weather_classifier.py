import tempfile
import unittest
import zipfile
from pathlib import Path

from weather_classifier import evaluate, generate_report, load_rows


class WeatherClassifierTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            {"Outlook": "Sunny", "Wind": "Weak", "Play": "Yes"},
            {"Outlook": "Sunny", "Wind": "Strong", "Play": "Yes"},
            {"Outlook": "Rain", "Wind": "Weak", "Play": "No"},
            {"Outlook": "Rain", "Wind": "Strong", "Play": "No"},
        ]

    def test_leave_one_out_predicts_separable_outcomes(self):
        classes, _, predictions, matrix, accuracy = evaluate(
            self.rows, ["Outlook", "Wind"], "Play"
        )
        self.assertEqual(classes, ["No", "Yes"])
        self.assertEqual(accuracy, 1.0)
        self.assertEqual(matrix["Yes"]["Yes"], 2)
        self.assertEqual(matrix["No"]["No"], 2)
        self.assertEqual(len(predictions), len(self.rows))

    def test_reads_docx_table(self):
        xml = (
            '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
            '<w:body><w:tbl>'
            '<w:tr><w:tc><w:p><w:r><w:t>Weather</w:t></w:r></w:p></w:tc>'
            '<w:tc><w:p><w:r><w:t>Play</w:t></w:r></w:p></w:tc></w:tr>'
            '<w:tr><w:tc><w:p><w:r><w:t>Sunny</w:t></w:r></w:p></w:tc>'
            '<w:tc><w:p><w:r><w:t>Yes</w:t></w:r></w:p></w:tc></w:tr>'
            '</w:tbl></w:body></w:document>'
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weather.docx"
            with zipfile.ZipFile(path, "w") as document:
                document.writestr("word/document.xml", xml)
            headers, rows = load_rows(path)
        self.assertEqual(headers, ["Weather", "Play"])
        self.assertEqual(rows, [{"Weather": "Sunny", "Play": "Yes"}])

    def test_generates_numbered_report_and_plots(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            accuracy = generate_report(output, self.rows, ["Outlook", "Wind"], "Play")
            report = (output / "report.md").read_text(encoding="utf-8")
            self.assertEqual(accuracy, 1.0)
            for section in ("## 1.", "## 2.", "## 3.", "## 4."):
                self.assertIn(section, report)
            for plot in ("outcome-distribution.svg", "confusion-matrix.svg",
                         "predictor-outcomes.svg"):
                self.assertTrue((output / "plots" / plot).is_file())


if __name__ == "__main__":
    unittest.main()
