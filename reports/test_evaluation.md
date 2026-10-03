# Sealed test evaluation — Formulation B

This is the one-shot score of the random forest chosen in `reports/tuning_results.md`.
No parameter was changed from these numbers. The unweighted forest is a pre-specified comparison, not a new search, and it does not replace the saved model.

## What was scored

- Sealed ids: **1319**.
- Unrated ids (NR/UR) excluded, because they are not on the ladder: **13**.
- Rows scored: **1306**.
- Training rows the models were fit on: **7402**.
- The training column below is resubstitution: the fitted model predicting the rows it was trained on. It is the memorisation check, not a second validation fold.

During tuning, the same forest's 5-fold training macro-F1 was 0.9742 ± 0.0015 and its validation macro-F1 was 0.5977 ± 0.0139.

## Saved model: random forest, class_weight=balanced_subsample

| metric | holdout | training rows, same fitted model |
|---|---:|---:|
| accuracy | 0.6363 | 0.9722 |
| macro-F1 | 0.5973 | 0.9714 |
| balanced accuracy | 0.5776 | 0.9744 |
| quadratic weighted kappa | 0.6546 | 0.9805 |
| ordinal MAE | 0.4280 | 0.0299 |
| adjacent accuracy | 0.9449 | 0.9980 |
| severe-error rate | 0.0551 | 0.0020 |
| Mature predicted as Kids | 4 | 0 |
| Kids predicted as Mature | 8 | 0 |

Holdout macro-F1 minus training macro-F1: **-0.3741**.
Top-label expected calibration error on the holdout: **0.0691**.

| class | precision | recall | F1 | support |
|---|---:|---:|---:|---:|
| Kids | 0.6812 | 0.5529 | 0.6104 | 85 |
| Older Kids | 0.5879 | 0.4350 | 0.5000 | 223 |
| Teens | 0.5556 | 0.5290 | 0.5419 | 397 |
| Mature | 0.6873 | 0.7937 | 0.7367 | 601 |

Confusion matrix, rows are the true bucket and columns are the prediction.

| true \ predicted | Kids | Older Kids | Teens | Mature |
|---|---:|---:|---:|---:|
| Kids | 47 | 30 | 0 | 8 |
| Older Kids | 18 | 97 | 61 | 47 |
| Teens | 0 | 25 | 210 | 162 |
| Mature | 4 | 13 | 107 | 477 |

| confidence bin | count | accuracy in bin | mean confidence |
|---|---:|---:|---:|
| 0.20–0.30 | 8 | 0.6250 | 0.2918 |
| 0.30–0.40 | 137 | 0.3431 | 0.3630 |
| 0.40–0.50 | 303 | 0.4686 | 0.4532 |
| 0.50–0.60 | 347 | 0.6225 | 0.5512 |
| 0.60–0.70 | 265 | 0.7736 | 0.6479 |
| 0.70–0.80 | 157 | 0.8535 | 0.7442 |
| 0.80–0.90 | 63 | 0.9206 | 0.8436 |
| 0.90–1.00 | 26 | 0.9231 | 0.9294 |

Figure: `reports/figures/confusion_matrix_test.png`.

## Pre-specified comparison: same forest, class_weight=None

| metric | holdout | training rows, same fitted model |
|---|---:|---:|
| accuracy | 0.6478 | 0.9614 |
| macro-F1 | 0.5991 | 0.9606 |
| balanced accuracy | 0.5641 | 0.9499 |
| quadratic weighted kappa | 0.6629 | 0.9606 |
| ordinal MAE | 0.4150 | 0.0476 |
| adjacent accuracy | 0.9449 | 0.9922 |
| severe-error rate | 0.0551 | 0.0078 |
| Mature predicted as Kids | 1 | 0 |
| Kids predicted as Mature | 9 | 8 |

Holdout macro-F1 minus training macro-F1: **-0.3615**.
Top-label expected calibration error on the holdout: **0.0424**.

| class | precision | recall | F1 | support |
|---|---:|---:|---:|---:|
| Kids | 0.8077 | 0.4941 | 0.6131 | 85 |
| Older Kids | 0.6622 | 0.4395 | 0.5283 | 223 |
| Teens | 0.5884 | 0.4358 | 0.5007 | 397 |
| Mature | 0.6564 | 0.8869 | 0.7544 | 601 |

Confusion matrix, rows are the true bucket and columns are the prediction.

| true \ predicted | Kids | Older Kids | Teens | Mature |
|---|---:|---:|---:|---:|
| Kids | 42 | 34 | 0 | 9 |
| Older Kids | 9 | 98 | 58 | 58 |
| Teens | 0 | 12 | 173 | 212 |
| Mature | 1 | 4 | 63 | 533 |

| confidence bin | count | accuracy in bin | mean confidence |
|---|---:|---:|---:|
| 0.30–0.40 | 59 | 0.3898 | 0.3697 |
| 0.40–0.50 | 249 | 0.4458 | 0.4615 |
| 0.50–0.60 | 348 | 0.5661 | 0.5503 |
| 0.60–0.70 | 302 | 0.6987 | 0.6481 |
| 0.70–0.80 | 202 | 0.8515 | 0.7446 |
| 0.80–0.90 | 103 | 0.9029 | 0.8392 |
| 0.90–1.00 | 43 | 0.9070 | 0.9384 |

Figure: `reports/figures/confusion_matrix_test_unweighted.png`.

Both reliability curves are in `reports/figures/calibration_test.png`.

## How to read the gap and the class weights

The saved forest's holdout macro-F1 is 0.5973 against a training-row macro-F1 of 0.9714. Holdout accuracy is 0.6363 against training-row accuracy 0.9722.

Removing class weights moves holdout accuracy from 0.6363 to 0.6478 and holdout macro-F1 from 0.5973 to 0.5991. Mature predicted as Kids moves from 4 to 1.

The saved model stays the balanced forest. This table is the measurement, not a reason to retune.

The same no-weight forest, scored on the training folds rather than this holdout, has accuracy 0.6348 and macro-F1 0.5765, against the balanced forest's out-of-fold accuracy 0.6308 and macro-F1 0.5981. The small holdout macro-F1 edge for the unweighted forest does not appear on the training folds. Detail: `reports/class_weight_oof.json`.
