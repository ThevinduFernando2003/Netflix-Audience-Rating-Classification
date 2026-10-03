# Findings

Figures are copied from the reports named in each section. Re-run the script that writes a report before changing a number here.

## The catalogue is not the brief's catalogue

`reports/data_profile.md` and `docs/DECISIONS.md`: 8,790 rows, 10 columns, no `cast`, no `description`, duplicate `show_id` count 0. `Not Given` appears 2,588 times in `director` and 287 times in `country`. Date and duration parse failures are 0. Contaminated ratings are 0. Missing ratings are 0.

`reports/target_design.md`: of 6,126 movies, 4,430 (72.3%) carry a TV certificate and 1,618 (26.4%) carry a film certificate. TV shows are almost entirely TV certificates (2,658) with 2 film certificates and 4 unrated. The two certification systems are stacked in one column. They are not separated by `type`.

## Formulation B is the product label

Pooled out-of-fold quadratic weighted kappa, training pool only:

| model | A | B | C |
|---|---:|---:|---:|
| logistic regression | 0.5969 | 0.6417 | 0.5792 |
| random forest | 0.5803 | 0.6049 | 0.5873 |
| histogram gradient boosting | 0.6142 | 0.6579 | 0.6213 |

Logistic regression was the pre-registered primary model and random forest the confirming model. Both rank B first. Histogram gradient boosting is stronger on B (quadratic weighted kappa 0.6579, macro-F1 0.5754, Mature predicted as Kids 32) but it was not the selection model. The rule was not "always B".

On Formulation B, logistic regression reaches accuracy 0.6000, macro-F1 0.5647, ordinal MAE 0.4789, adjacent accuracy 0.9287, Mature predicted as Kids 38, and Kids predicted as Mature 18. The same model's Mature-as-Kids counts are 117 on A and 125 on C. The most-frequent dummy on B has accuracy 0.4601 and macro-F1 0.1576, which is what predicting Mature for every row is worth.

Training-pool class counts for B: Mature 3,406, Teens 2,250, Older Kids 1,264, Kids 482. Kids is the rare class. That is why macro-F1, not accuracy, is the later selection metric.

## No feature restates the certificate

On the 7,402 Formulation B training rows, label entropy is 1.1988 nats. The strongest feature, `genre__Children & Family Movies`, has mutual information 0.1008 nats, 0.084 of that entropy. Nothing reaches 0.90. Genre is a content descriptor. It is not a copy of the rating.

## The tuned forest wins macro-F1 and memorizes

Full-pool 5-fold scores of each family's best parameters (`reports/tuning_results.md`):

| model | macro-F1 | balanced accuracy | QWK | train macro-F1 |
|---|---:|---:|---:|---:|
| random forest | 0.5977 ± 0.0139 | 0.5830 ± 0.0144 | 0.6596 ± 0.0178 | 0.9742 ± 0.0015 |
| histogram gradient boosting | 0.5768 ± 0.0130 | 0.5825 ± 0.0176 | 0.6620 ± 0.0229 | 0.8669 ± 0.0065 |
| logistic regression | 0.5576 ± 0.0155 | 0.5861 ± 0.0214 | 0.6426 ± 0.0284 | 0.6462 ± 0.0072 |
| decision tree | 0.4938 ± 0.0223 | 0.5319 ± 0.0164 | 0.5556 ± 0.0353 | 0.5442 ± 0.0166 |

The forest has the highest validation macro-F1 and the widest train/validation gap. Logistic regression generalises more tightly and loses on macro-F1. Histogram gradient boosting has a slightly higher quadratic weighted kappa (0.6620 against 0.6596) and a lower macro-F1, so it was not selected. The validation curve for the forest shows the train/validation macro-F1 gap widest at unlimited depth (0.429).

## The holdout matches the cross-validation estimate

`reports/test_evaluation.md`, saved forest, 1,306 labelled titles:

| metric | holdout | training rows |
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

Holdout macro-F1 0.5973 sits on the cross-validation estimate 0.5977 ± 0.0139. The memorisation gap is the resubstitution gap, not a surprise collapse on the holdout. Top-label expected calibration error is 0.0691. In the 0.30–0.40 confidence bin, accuracy is 0.3431 against mean confidence 0.3630 (137 titles). In the 0.70–0.80 bin, accuracy is 0.8535 against mean confidence 0.7442 (157 titles).

Per-class holdout F1: Kids 0.6104 (support 85), Older Kids 0.5000 (223), Teens 0.5419 (397), Mature 0.7367 (601). The confusion matrix shows the mass of errors on the adjacent pairs Teens/Mature and Kids/Older Kids. Of 601 Mature titles, 4 are predicted Kids, 13 Older Kids, 107 Teens, and 477 Mature.

## Dropping class weights does not win both metrics

On the holdout, the pre-specified unweighted twin has accuracy 0.6478 and macro-F1 0.5991, with Mature predicted as Kids equal to 1 and expected calibration error 0.0424. That holdout edge is small next to the cross-validation standard deviation of macro-F1 (0.0139).

On the training folds the same unweighted forest has accuracy 0.6348 and macro-F1 0.5765, against the balanced forest's 0.6308 and 0.5981. Kids predicted as Mature rises to 71. Quadratic weighted kappa falls from 0.6597 to 0.6294. The training folds do not support dropping the weights. The saved model stays `class_weight=balanced_subsample`.

## A vote can raise ordinal agreement without raising both accuracy and macro-F1

Out-of-fold, the balanced forest alone is accuracy 0.6308 and macro-F1 0.5981. The pre-registered mix with weights logistic regression 1, forest 2, boosting 1 reaches accuracy 0.6298, macro-F1 0.5993, and quadratic weighted kappa 0.6780, with Mature predicted as Kids equal to 12 against the forest's 6. Macro-F1 and quadratic weighted kappa rise. Accuracy does not. The adoption rule required both accuracy and macro-F1, so the mix was not adopted and was not scored on the holdout.

Multiplying the Mature probability by 0.85 reaches out-of-fold macro-F1 0.6002 and accuracy 0.6244. Again macro-F1 rises and accuracy falls. No pre-registered scale was adopted.

## What this model is good for

Adjacent accuracy on the holdout is 0.9449, so about 19 in 20 predictions are exact or one bucket away. The serious failure, Mature predicted as Kids, is rare on the holdout (4 of 601) and is still the case the API warns about when the prediction itself is Kids. The model is not a substitute for the certificate. Training macro-F1 0.9714 against holdout macro-F1 0.5973 means a title the forest has already seen will look solved, and a new title will not.
