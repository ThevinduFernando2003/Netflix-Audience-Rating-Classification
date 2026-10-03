# Training-fold vote — Formulation B

The sealed test set was not loaded. Weights below were fixed before these probabilities were computed.

Adopt a pre-registered weight mix only when out-of-fold accuracy and out-of-fold macro-F1 both exceed the random forest alone. Otherwise keep the saved forest. The sealed test set is not an input.

Adopted: **False**.

| logistic regression | random forest | hist gradient boosting | accuracy | macro-F1 | balanced accuracy | QWK | Mature→Kids |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0 | 0 | 0.5975 | 0.5580 | 0.5861 | 0.6428 | 46 |
| 0 | 1 | 0 | 0.6308 | 0.5981 | 0.5831 | 0.6597 | 6 |
| 0 | 0 | 1 | 0.6088 | 0.5773 | 0.5825 | 0.6623 | 21 |
| 1 | 1 | 1 | 0.6263 | 0.5938 | 0.5962 | 0.6766 | 14 |
| 1 | 2 | 1 | 0.6298 | 0.5993 | 0.5964 | 0.6780 | 12 |
| 2 | 1 | 1 | 0.6205 | 0.5861 | 0.5954 | 0.6704 | 20 |
| 1 | 1 | 2 | 0.6238 | 0.5916 | 0.5942 | 0.6757 | 15 |
| 1 | 2 | 2 | 0.6271 | 0.5941 | 0.5932 | 0.6775 | 12 |

Random forest alone, out of fold: accuracy 0.6308, macro-F1 0.5981.
Highest macro-F1 candidate, not adopted: logistic regression 1, random forest 2, hist gradient boosting 1, accuracy 0.6298, macro-F1 0.5993, QWK 0.6780.

No pre-registered mix beat the forest on both accuracy and macro-F1. The saved forest stays the model. The holdout was not scored again for this search.

## Class-probability scales

Eight pre-registered scales were applied to the cached out-of-fold probabilities. The best macro-F1 was 0.6002, from multiplying the Mature probability by 0.85, and its accuracy was 0.6244. That is a higher macro-F1 and a lower accuracy than the forest. None of the scales beat the forest on both, so none was adopted. Detail is in `reports/threshold_decision.json`.

## Class weight on the same training folds

`class_weight=None` on the same forest and the same folds: accuracy 0.6348, macro-F1 0.5765, QWK 0.6294, Mature predicted as Kids 0, Kids predicted as Mature 71.

The balanced forest on those folds: accuracy 0.6308, macro-F1 0.5981, QWK 0.6597, Mature predicted as Kids 6.

No class weight raises accuracy and lowers macro-F1. It is not adopted. The holdout comparison in `reports/test_evaluation.md` was already measured and was not used for this choice.
