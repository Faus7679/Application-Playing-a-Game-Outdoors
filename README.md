# Application-Playing-a-Game-Outdoors

This project builds and evaluates a categorical Naive Bayes classifier to recommend
whether an outdoor game should be played from historical weather observations.
The referenced `C:\Users\Yawo Faustin AZIAKPO\Downloads\CST-570-RS-WeatherDataSet.docx`
is on the original author's computer and is not included in this repository. No
dataset values or model results have therefore been assumed or fabricated.

## Benchmark analysis (illustrative only)

`weather_naive_bayes.py` runs a categorical Naive Bayes classifier with add-one
smoothing and leave-one-out cross-validation on the commonly published 14-row
Play Tennis benchmark. **This is not the course dataset** and its results are not an
evaluation of the missing DOCX. To analyse the course data, replace the rows in
`ROWS` (and adapt `SCHEMA` if needed) with the actual labeled observations.

```sh
python weather_naive_bayes.py
```

The command prints per-fold predictions and aggregate metrics (LOOCV accuracy 7/14 =
50.0%, TP=6 FN=3 FP=4 TN=1, majority baseline 9/14 = 64.3%) and writes four SVG plots to
`plots/`: `class_priors_and_smoothing.svg`, `loocv_predictions.svg`,
`confusion_and_baseline.svg` and `feature_outcome_distribution.svg`. The write-up that
references them is in [`report.md`](report.md). It makes no safety guarantee.

## Generic CSV/DOCX analysis

The input can be a CSV file or a DOCX containing a Word table. The first table row
must contain unique column names; each subsequent row must contain one observation.
Weather predictors are treated as categorical, and the outcome column must have at
least two non-empty classes. The script uses only the Python standard library.

```sh
python weather_classifier.py \
  --data /path/to/CST-570-RS-WeatherDataSet.docx \
  --target Play \
  --output weather-analysis
```

Replace `Play` with the exact name of the dataset's outcome column. Alternatively,
export the data table to CSV and pass that file to `--data`. The command creates
`weather-analysis/report.md` and three SVG plots in `weather-analysis/plots/`.
The report contains numbered answers covering the model, training and validation,
LOOCV accuracy, and assumptions and limitations. It reports results only after
running on the supplied data.

The model uses add-one (Laplace) smoothing by default. `--alpha` changes the
positive smoothing parameter. Leave-one-out cross-validation (LOOCV) holds out
one historical observation at a time, fits on the others, and summarizes all
held-out predictions. The resulting confusion matrix and accuracy describe
validation performance; a final model for use should be refit on all observations.
Validation on new data is preferable when a separate representative dataset is
available.

## Tests

```sh
python -m unittest discover -s tests
```

## References

Domingos, P., & Pazzani, M. (1997). On the optimality of the simple Bayesian
classifier under zero-one loss. *Machine Learning, 29*, 103–130.
https://doi.org/10.1023/A:1007413511361

Zhang, H. (2004). The optimality of Naive Bayes. In *Proceedings of the
Seventeenth International Florida Artificial Intelligence Research Society
Conference* (pp. 562–567).
