# Naive Bayes classifier for playing a game outdoors

This analysis uses the 14 labeled observations in the CST-570 Weather DataSet
supplied in the assignment prompt. Those table values are the source dataset; no
separate course document was used. The predictors are `weather`, `temperature`,
`humidity`, and `windy`; the target is `play` (`yes` or `no`). The data, model, and
reproducible analysis are defined in `weather_naive_bayes.py`.

## 1. Naive Bayes model

For an observation $x=(x_1,\ldots,x_4)$, categorical Naive Bayes predicts the
class with the largest posterior score:

$$\hat{c}=\arg\max_{c\in\{yes,no\}} P(c)\prod_{j=1}^{4}P(x_j\mid c).$$

Class priors are estimated from the training-set class frequencies. Conditional
probabilities use Laplace (add-one) smoothing:

$$P(x_j=v\mid c)=\frac{N_{j,v,c}+1}{N_c+K_j},$$

where $N_{j,v,c}$ is the count of feature value $v$ in class $c$, $N_c$ is the
number of training rows in class $c$, and $K_j$ is the number of categories in
the declared feature domain. The implementation compares log probabilities to
avoid numerical underflow.

The complete dataset has 9 `yes` and 5 `no` labels, giving unsmoothed priors
$P(yes)=9/14$ and $P(no)=5/14$. For example, among the 9 `yes` rows, `weather`
is `sunny` twice. As `weather` has three declared values, the smoothed example
is $P(weather=sunny\mid yes)=(2+1)/(9+3)=1/4$.

![Class counts, priors and smoothed example probability](plots/class_priors_and_smoothing.svg)

## 2. Training and validation

The model counts class and feature-value frequencies in each training set. It
uses leave-one-out cross-validation (LOOCV): each of the 14 rows is held out
once, the model is fitted on the other 13 rows, and that held-out row is
predicted. The schema domains are fixed from the supplied dataset and do not
depend on the held-out row.

Run the reproducible analysis and regenerate the charts with:

```sh
python weather_naive_bayes.py
```

The fold-by-fold predictions are:

| Row | Weather | Temperature | Humidity | Windy | Actual | Predicted |
|---:|---|---|---|---|---|---|
| 1 | sunny | above average | low | calm | no | yes |
| 2 | sunny | above average | low | windy | no | no |
| 3 | overcast | above average | low | calm | yes | yes |
| 4 | rainy | average | low | calm | yes | yes |
| 5 | rainy | average | high | calm | yes | yes |
| 6 | rainy | below average | high | windy | no | yes |
| 7 | overcast | below average | high | windy | yes | yes |
| 8 | sunny | below average | low | calm | no | yes |
| 9 | sunny | average | high | calm | yes | yes |
| 10 | rainy | below average | high | calm | yes | yes |
| 11 | sunny | average | low | windy | yes | no |
| 12 | overcast | average | low | windy | yes | yes |
| 13 | overcast | above average | high | calm | yes | yes |
| 14 | rainy | average | low | windy | no | yes |

![LOOCV fold predictions](plots/loocv_predictions.svg)

## 3. Accuracy

LOOCV correctly predicts 9 of 14 labels:

$$\text{accuracy}=\frac{\text{correct predictions}}{\text{all predictions}}
=\frac{9}{14}\approx 0.6429=64.3\%.$$

Taking `yes` as the positive class, the confusion counts are TP = 8, FN = 1,
FP = 4, and TN = 1. The majority-class baseline predicts `yes` for every row
and is correct on 9/14 = 64.3%. Thus the LOOCV accuracy ties, rather than
exceeds, this baseline.

![Confusion matrix and accuracy versus majority baseline](plots/confusion_and_baseline.svg)

![Outcome distribution by feature category](plots/feature_outcome_distribution.svg)

## 4. Assumptions and limitations

- **Retrospective label prediction, not a safety recommendation:** the model
  predicts the recorded historical `play` label. That label is not necessarily
  an indication that conditions were safe. This analysis does not recommend
  whether a game should actually be played.
- **Conditional independence:** given the `play` class, the model treats
  `weather`, `temperature`, `humidity`, and `windy` as independent. Real
  weather variables can be correlated, so multiplying their separate
  conditional probabilities may misstate the joint probability.
- **Small sample and validation uncertainty:** 14 observations provide little
  evidence about generalization. LOOCV reuses nearly all rows for training in
  every fold, but its estimate can still have high variance and is not a
  substitute for validation on representative future data.
- **Categorical encoding and schema:** the recorded categories are treated as
  categorical values with the domains stated above. Different category
  definitions or continuous measurements would require justified preprocessing
  or a different model.
- **Smoothing and decision costs:** add-one smoothing avoids zero conditional
  probabilities but may not be optimal. Accuracy weighs all errors equally and
  does not represent the consequences of unsafe play versus unnecessary
  cancellation.
- **No safety guarantee:** a real decision requires current local forecasts,
  official severe-weather alerts, appropriate safeguards, and human judgment.

## References

Domingos, P., & Pazzani, M. (1997). On the optimality of the simple Bayesian classifier
under zero-one loss. *Machine Learning, 29*, 103–130.
https://doi.org/10.1023/A:1007413511361

Mitchell, T. M. (1997). *Machine learning*. McGraw-Hill.

Witten, I. H., Frank, E., & Hall, M. A. (2011). *Data mining: Practical machine
learning tools and techniques* (3rd ed.). Morgan Kaufmann.
