# Target design

## Product choice: Formulation B — maturity buckets

- logistic_regression pooled QWK ranks B first or alone at the top (0.6417; A=0.5969, C=0.5792).
- random_forest pooled QWK agrees within 0.02: A=0.5803, B=0.6049, C=0.5873.
- Native accuracy is not the selection metric. The largest bucket is almost half the rows, so accuracy mostly restates the majority class. Dummy most-frequent is in the tables for that reason.

Primary comparison model: `logistic_regression`. Confirming model: `random_forest`. Tie margin: 0.02 pooled out-of-fold quadratic weighted kappa (QWK). The sealed 15% test set was not scored. Figures below are cross-validated on the training pool only.

Pooled QWK used by the rule:

| model | A | B | C |
|---|---:|---:|---:|
| `logistic_regression` | 0.5969 | 0.6417 | 0.5792 |
| `random_forest` | 0.5803 | 0.6049 | 0.5873 |

## What was compared

Formulation A keeps raw certificates and folds any training-pool class with fewer than 5 rows into `OTHER`. If that leftover class still cannot support 5-fold stratification, those rows are dropped and the count is below.

Formulation B maps both certification systems onto one 4-level ordinal scale. The map lives in `config.yaml` under `maturity_buckets`.

Formulation C fits one model for movies and one model for TV shows, routed by `type`. It is not a clean film-certificate model plus a clean TV-certificate model. On this file, most movies are labelled with a TV certificate, so a film-only movie model would discard them.

### Certification system by type

Counts are the full catalogue, including the sealed test rows. This table is a description of the labels, not a model score.

| type | film | tv | unrated |
|---|---:|---:|---:|
| Movie | 1618 | 4430 | 78 |
| TV Show | 2 | 2658 | 4 |

Movies: 6126 titles, of which 4430 (72.3%) carry a TV certificate and 1618 (26.4%) carry a film certificate. A movie model restricted to film certificates would ignore the majority of movies. That is why Formulation C is routed by `type` and trained on the certificates each type actually uses.

## Where each certificate sits, and why

The ladder used as a guide is `TV-Y < TV-G < TV-Y7 < TV-PG < PG < TV-14 < PG-13 < R < TV-MA < NC-17`. `G` is the film certificate next to `TV-G`. Buckets follow that order. They do not follow alphabetical order.

- **Kids (0)** — `TV-Y`, `TV-G`, `G`. All three are unrestricted: young-child TV, general TV, and general film. A parent does not need a warning.
- **Older Kids (1)** — `TV-Y7`, `TV-Y7-FV`, `PG`, `TV-PG`. `TV-Y7` is aimed at children over 7. `TV-Y7-FV` is that same certificate with a fantasy-violence flag (6 titles), not a separate maturity level. `TV-PG` and `PG` are the judgement call: the ladder puts `TV-PG` just below `PG`, and both mean parental guidance rather than a teen restriction. They sit next to each other, below `TV-14` / `PG-13`, so they share a bucket. Splitting them would rebuild the two-certificate problem inside one boundary.
- **Teens (2)** — `PG-13`, `TV-14`. Both are the 'not for younger children' band. The ladder puts `TV-14` just below `PG-13`. They are one product decision apart, not a different audience.
- **Mature (3)** — `R`, `NC-17`, `TV-MA`. `R` and `TV-MA` are the adult certificates. `NC-17` is stricter than `R` and has 3 titles, which cannot support its own class under 5-fold stratification. It stays at the top of the ladder instead of being dropped or called a teen rating.
- **`NR` and `UR` are not on the ladder.** They mean the title was not certified. Formulation B drops them. The count is in the exclusions section.

## Rows excluded from a formulation

- Contaminated `rating` (duration string): **0**. Excluded from every formulation.
- Missing `rating`: **0**. Excluded from every formulation. Kept as the unlabelled demo pool.
- Sealed test rows, not scored: **1319**.
- Training-pool rows available to the comparison: **7471**.
- Formulation B drops unrated (`NR`, `UR`) rows inside the training pool: **69**.

### Formulation A label stabilisation (training pool)

- Folded into `OTHER`: `UR`=3, `NC-17`=1.
- `OTHER` rows dropped because the folded class still had fewer than 5 rows: **4**.
- Other classes dropped for the same reason: (none).

### Formulation C label stabilisation (training pool, per type)

**Movie** — rows before the unrated filter: 5202; unrated dropped: 65.

- Folded into `OTHER`: `NC-17`=1.
- Folded class dropped: **1**.
- Still too small and dropped: (none).

**TV Show** — rows before the unrated filter: 2269; unrated dropped: 4.

- Folded into `OTHER`: `R`=2, `TV-Y7-FV`=1.
- Folded class dropped: **3**.
- Still too small and dropped: (none).

## Class counts in the training pool after stabilisation

### Formulation A

| label | rows |
|---|---:|
| `TV-MA` | 2709 |
| `TV-14` | 1842 |
| `TV-PG` | 731 |
| `R` | 696 |
| `PG-13` | 408 |
| `TV-Y7` | 288 |
| `TV-Y` | 255 |
| `PG` | 239 |
| `TV-G` | 191 |
| `NR` | 66 |
| `G` | 36 |
| `TV-Y7-FV` | 6 |

### Formulation B

| label | rows |
|---|---:|
| `Mature` | 3406 |
| `Teens` | 2250 |
| `Older Kids` | 1264 |
| `Kids` | 482 |

### Formulation C

**Movie**

| label | rows |
|---|---:|
| `TV-MA` | 1740 |
| `TV-14` | 1220 |
| `R` | 694 |
| `TV-PG` | 455 |
| `PG-13` | 408 |
| `PG` | 239 |
| `TV-Y7` | 122 |
| `TV-Y` | 111 |
| `TV-G` | 106 |
| `G` | 36 |
| `TV-Y7-FV` | 5 |

**TV Show**

| label | rows |
|---|---:|
| `TV-MA` | 969 |
| `TV-14` | 622 |
| `TV-PG` | 276 |
| `TV-Y7` | 166 |
| `TV-Y` | 144 |
| `TV-G` | 85 |

## Protocol

Seed `42`. Stratified holdout `15%` using the bucket, with unrated rows as their own stratum so the split does not depend on a 3-row class. Model comparison uses 5-fold `StratifiedKFold(shuffle=True)` on each formulation's own training labels. Formulation C runs that split separately inside movies and inside TV shows.

Every vectorizer, imputer, scaler, and one-hot encoder is fit on the training side of the fold only. Title TF-IDF, genre indicators, type, primary country, primary genre, numeric metadata, and the lexicon counts are shared. Description TF-IDF is omitted because that column is empty. `ComplementNB` uses a non-negative branch (median impute, no standardisation, clipped at zero) because standardisation produces negatives and `years_to_platform` can be negative.

Hyperparameters are the fixed comparison settings in `config.yaml` under `comparison.estimators`. They are not tuned. Tuning the winner is the next stage. Total comparison wall clock: **1543.2 seconds**.

Ordinal metrics map predictions back onto the 4-level ladder. `OTHER`, `NR`, and `UR` predictions are not given a fake level: they count as unmapped. QWK uses weights `(i-j)^2 / 9` on levels 0–3. `mature_as_kids` is the safety-critical error: a Mature title predicted as Kids.

For Formulation C, QWK, ordinal MAE, adjacent accuracy, severe-error rate, and the Kids/Mature counts are scored on the concatenated out-of-fold predictions, on the shared 4-level ladder. Macro-F1 and balanced accuracy are the support-weighted means of the movie model and the TV model: those metrics are not meaningful on the union of two different certificate sets. Routed accuracy equals that same weighted mean. Per-type mean ± std follows.

HistGradientBoosting uses `max_iter=50` and `max_bins=64`. The library default of 255 bins on the dense title matrix was still running after 25 minutes for one formulation, so the comparison budget was reduced and written down in `config.yaml` and `docs/DECISIONS.md`. These are not tuned parameters.

## Pooled out-of-fold results

### QWK

| model | A | B | C |
|---|---:|---:|---:|
| `dummy_most_frequent` | 0.0000 | 0.0000 | 0.0000 |
| `dummy_stratified` | 0.0160 | 0.0038 | 0.0107 |
| `logistic_regression` | 0.5969 | 0.6417 | 0.5792 |
| `decision_tree` | 0.4996 | 0.5565 | 0.5146 |
| `random_forest` | 0.5803 | 0.6049 | 0.5873 |
| `extra_trees` | 0.5882 | 0.5975 | 0.5698 |
| `hist_gradient_boosting` | 0.6142 | 0.6579 | 0.6213 |
| `complement_nb` | 0.4752 | 0.5607 | 0.4982 |

### Macro-F1 (native label space; C is the support-weighted mean of the two models)

| model | A | B | C |
|---|---:|---:|---:|
| `dummy_most_frequent` | 0.0444 | 0.1576 | 0.0625 |
| `dummy_stratified` | 0.0848 | 0.2520 | 0.1087 |
| `logistic_regression` | 0.3961 | 0.5647 | 0.4419 |
| `decision_tree` | 0.3338 | 0.5005 | 0.3834 |
| `random_forest` | 0.3547 | 0.5284 | 0.4284 |
| `extra_trees` | 0.3436 | 0.5248 | 0.4243 |
| `hist_gradient_boosting` | 0.4065 | 0.5754 | 0.4430 |
| `complement_nb` | 0.2853 | 0.5142 | 0.3435 |

### Accuracy

| model | A | B | C |
|---|---:|---:|---:|
| `dummy_most_frequent` | 0.3628 | 0.4601 | 0.3662 |
| `dummy_stratified` | 0.2228 | 0.3453 | 0.2264 |
| `logistic_regression` | 0.4867 | 0.6000 | 0.5149 |
| `decision_tree` | 0.4225 | 0.5324 | 0.4423 |
| `random_forest` | 0.4480 | 0.5763 | 0.5065 |
| `extra_trees` | 0.4282 | 0.5684 | 0.4931 |
| `hist_gradient_boosting` | 0.5165 | 0.6131 | 0.5364 |
| `complement_nb` | 0.5078 | 0.5798 | 0.5138 |

### Balanced accuracy

| model | A | B | C |
|---|---:|---:|---:|
| `dummy_most_frequent` | 0.0833 | 0.2500 | 0.1141 |
| `dummy_stratified` | 0.0850 | 0.2521 | 0.1093 |
| `logistic_regression` | 0.4662 | 0.5892 | 0.4831 |
| `decision_tree` | 0.3943 | 0.5230 | 0.4279 |
| `random_forest` | 0.4409 | 0.5617 | 0.4688 |
| `extra_trees` | 0.4214 | 0.5647 | 0.4751 |
| `hist_gradient_boosting` | 0.4251 | 0.5872 | 0.4512 |
| `complement_nb` | 0.2938 | 0.5057 | 0.3500 |

### Ordinal MAE

| model | A | B | C |
|---|---:|---:|---:|
| `dummy_most_frequent` | 0.8410 | 0.8409 | 0.8410 |
| `dummy_stratified` | 0.9608 | 0.9689 | 0.9701 |
| `logistic_regression` | 0.5231 | 0.4789 | 0.5377 |
| `decision_tree` | 0.6329 | 0.5680 | 0.6311 |
| `random_forest` | 0.5338 | 0.5166 | 0.5253 |
| `extra_trees` | 0.5267 | 0.5269 | 0.5465 |
| `hist_gradient_boosting` | 0.5007 | 0.4550 | 0.4842 |
| `complement_nb` | 0.5522 | 0.5218 | 0.5414 |

### Adjacent accuracy (within 1 level)

| model | A | B | C |
|---|---:|---:|---:|
| `dummy_most_frequent` | 0.7641 | 0.7641 | 0.7641 |
| `dummy_stratified` | 0.7490 | 0.7466 | 0.7493 |
| `logistic_regression` | 0.9110 | 0.9287 | 0.9020 |
| `decision_tree` | 0.8757 | 0.9138 | 0.8804 |
| `random_forest` | 0.9158 | 0.9235 | 0.9121 |
| `extra_trees` | 0.9217 | 0.9223 | 0.9074 |
| `hist_gradient_boosting` | 0.9239 | 0.9389 | 0.9300 |
| `complement_nb` | 0.8866 | 0.9123 | 0.8929 |

### Severe-error rate (off by 2 or more levels, mapped predictions only)

| model | A | B | C |
|---|---:|---:|---:|
| `dummy_most_frequent` | 0.2359 | 0.2359 | 0.2359 |
| `dummy_stratified` | 0.2510 | 0.2534 | 0.2507 |
| `logistic_regression` | 0.0890 | 0.0713 | 0.0980 |
| `decision_tree` | 0.1243 | 0.0862 | 0.1196 |
| `random_forest` | 0.0842 | 0.0765 | 0.0879 |
| `extra_trees` | 0.0783 | 0.0777 | 0.0926 |
| `hist_gradient_boosting` | 0.0761 | 0.0611 | 0.0700 |
| `complement_nb` | 0.1134 | 0.0877 | 0.1071 |

### Unmapped prediction rate

| model | A | B | C |
|---|---:|---:|---:|
| `dummy_most_frequent` | 0.0000 | 0.0000 | 0.0000 |
| `dummy_stratified` | 0.0068 | 0.0000 | 0.0000 |
| `logistic_regression` | 0.0486 | 0.0000 | 0.0000 |
| `decision_tree` | 0.0468 | 0.0000 | 0.0000 |
| `random_forest` | 0.0708 | 0.0000 | 0.0000 |
| `extra_trees` | 0.1065 | 0.0000 | 0.0000 |
| `hist_gradient_boosting` | 0.0123 | 0.0000 | 0.0000 |
| `complement_nb` | 0.0000 | 0.0000 | 0.0000 |

### Mature predicted as Kids (safety-critical count)

| model | A | B | C |
|---|---:|---:|---:|
| `dummy_most_frequent` | 0 | 0 | 0 |
| `dummy_stratified` | 180 | 231 | 223 |
| `logistic_regression` | 117 | 38 | 125 |
| `decision_tree` | 149 | 79 | 134 |
| `random_forest` | 160 | 101 | 94 |
| `extra_trees` | 157 | 111 | 127 |
| `hist_gradient_boosting` | 46 | 32 | 29 |
| `complement_nb` | 12 | 50 | 14 |

### Kids predicted as Mature (count)

| model | A | B | C |
|---|---:|---:|---:|
| `dummy_most_frequent` | 482 | 482 | 482 |
| `dummy_stratified` | 210 | 219 | 200 |
| `logistic_regression` | 15 | 18 | 15 |
| `decision_tree` | 41 | 26 | 16 |
| `random_forest` | 17 | 21 | 16 |
| `extra_trees` | 7 | 19 | 9 |
| `hist_gradient_boosting` | 14 | 20 | 23 |
| `complement_nb` | 160 | 53 | 135 |

## 5-fold mean ± std

### Formulation A — QWK

| model | mean ± std | folds |
|---|---:|---:|
| `dummy_most_frequent` | 0.0000 ± 0.0000 | 5 |
| `dummy_stratified` | 0.0160 ± 0.0210 | 5 |
| `logistic_regression` | 0.5967 ± 0.0219 | 5 |
| `decision_tree` | 0.5002 ± 0.0168 | 5 |
| `random_forest` | 0.5804 ± 0.0270 | 5 |
| `extra_trees` | 0.5888 ± 0.0400 | 5 |
| `hist_gradient_boosting` | 0.6143 ± 0.0176 | 5 |
| `complement_nb` | 0.4753 ± 0.0267 | 5 |

### Formulation A — macro-F1

| model | mean ± std | folds |
|---|---:|---:|
| `dummy_most_frequent` | 0.0444 ± 0.0000 | 5 |
| `dummy_stratified` | 0.0848 ± 0.0062 | 5 |
| `logistic_regression` | 0.3960 ± 0.0234 | 5 |
| `decision_tree` | 0.3339 ± 0.0151 | 5 |
| `random_forest` | 0.3551 ± 0.0085 | 5 |
| `extra_trees` | 0.3437 ± 0.0168 | 5 |
| `hist_gradient_boosting` | 0.4050 ± 0.0165 | 5 |
| `complement_nb` | 0.2852 ± 0.0143 | 5 |

### Formulation B — QWK

| model | mean ± std | folds |
|---|---:|---:|
| `dummy_most_frequent` | 0.0000 ± 0.0000 | 5 |
| `dummy_stratified` | 0.0038 ± 0.0279 | 5 |
| `logistic_regression` | 0.6414 ± 0.0284 | 5 |
| `decision_tree` | 0.5566 ± 0.0241 | 5 |
| `random_forest` | 0.6050 ± 0.0258 | 5 |
| `extra_trees` | 0.5979 ± 0.0313 | 5 |
| `hist_gradient_boosting` | 0.6575 ± 0.0259 | 5 |
| `complement_nb` | 0.5610 ± 0.0285 | 5 |

### Formulation B — macro-F1

| model | mean ± std | folds |
|---|---:|---:|
| `dummy_most_frequent` | 0.1576 ± 0.0001 | 5 |
| `dummy_stratified` | 0.2520 ± 0.0062 | 5 |
| `logistic_regression` | 0.5642 ± 0.0200 | 5 |
| `decision_tree` | 0.5001 ± 0.0173 | 5 |
| `random_forest` | 0.5283 ± 0.0146 | 5 |
| `extra_trees` | 0.5254 ± 0.0204 | 5 |
| `hist_gradient_boosting` | 0.5752 ± 0.0107 | 5 |
| `complement_nb` | 0.5141 ± 0.0197 | 5 |

### Formulation B — balanced accuracy

| model | mean ± std | folds |
|---|---:|---:|
| `dummy_most_frequent` | 0.2500 ± 0.0000 | 5 |
| `dummy_stratified` | 0.2521 ± 0.0062 | 5 |
| `logistic_regression` | 0.5892 ± 0.0247 | 5 |
| `decision_tree` | 0.5229 ± 0.0222 | 5 |
| `random_forest` | 0.5616 ± 0.0170 | 5 |
| `extra_trees` | 0.5647 ± 0.0173 | 5 |
| `hist_gradient_boosting` | 0.5872 ± 0.0151 | 5 |
| `complement_nb` | 0.5057 ± 0.0222 | 5 |

### Formulation C per type — QWK mean ± std

**Movie — QWK**

| model | mean ± std | folds |
|---|---:|---:|
| `dummy_most_frequent` | 0.0000 ± 0.0000 | 5 |
| `dummy_stratified` | -0.0022 ± 0.0289 | 5 |
| `logistic_regression` | 0.5403 ± 0.0211 | 5 |
| `decision_tree` | 0.4961 ± 0.0240 | 5 |
| `random_forest` | 0.5368 ± 0.0317 | 5 |
| `extra_trees` | 0.5192 ± 0.0267 | 5 |
| `hist_gradient_boosting` | 0.5890 ± 0.0211 | 5 |
| `complement_nb` | 0.3942 ± 0.0266 | 5 |

**Movie — macro-F1**

| model | mean ± std | folds |
|---|---:|---:|
| `dummy_most_frequent` | 0.0460 ± 0.0000 | 5 |
| `dummy_stratified` | 0.0857 ± 0.0061 | 5 |
| `logistic_regression` | 0.4136 ± 0.0188 | 5 |
| `decision_tree` | 0.3659 ± 0.0159 | 5 |
| `random_forest` | 0.4048 ± 0.0189 | 5 |
| `extra_trees` | 0.4013 ± 0.0138 | 5 |
| `hist_gradient_boosting` | 0.4346 ± 0.0182 | 5 |
| `complement_nb` | 0.3015 ± 0.0173 | 5 |

**TV Show — QWK**

| model | mean ± std | folds |
|---|---:|---:|
| `dummy_most_frequent` | 0.0000 ± 0.0000 | 5 |
| `dummy_stratified` | 0.0069 ± 0.0637 | 5 |
| `logistic_regression` | 0.6432 ± 0.0326 | 5 |
| `decision_tree` | 0.5315 ± 0.0350 | 5 |
| `random_forest` | 0.6762 ± 0.0453 | 5 |
| `extra_trees` | 0.6614 ± 0.0432 | 5 |
| `hist_gradient_boosting` | 0.6670 ± 0.0438 | 5 |
| `complement_nb` | 0.6439 ± 0.0400 | 5 |

**TV Show — macro-F1**

| model | mean ± std | folds |
|---|---:|---:|
| `dummy_most_frequent` | 0.1000 ± 0.0001 | 5 |
| `dummy_stratified` | 0.1610 ± 0.0215 | 5 |
| `logistic_regression` | 0.5007 ± 0.0196 | 5 |
| `decision_tree` | 0.4201 ± 0.0173 | 5 |
| `random_forest` | 0.4830 ± 0.0183 | 5 |
| `extra_trees` | 0.4775 ± 0.0213 | 5 |
| `hist_gradient_boosting` | 0.4626 ± 0.0219 | 5 |
| `complement_nb` | 0.4362 ± 0.0119 | 5 |

## Mutual information

Computed on the Formulation B training pool only (7402 rows). Label entropy is 1.1988 nats. The strongest single feature is `genre__Children & Family Movies` (genre) with MI 0.1008 nats, 0.084 of the label entropy.

No numeric feature, genre indicator, or title token reaches 0.90 of the label entropy. Nothing in the feature set restates the certificate.

No description TF-IDF was fitted. The description column is empty on every row of this extract, so a vectorizer would have an empty vocabulary. Title-token MI uses tokens with document frequency at least 5, capped at the 924 highest-frequency of those tokens.

| feature | kind | MI (nats) | share of label entropy |
|---|---|---:|---:|
| `genre__Children & Family Movies` | genre | 0.1008 | 0.084 |
| `genre__Kids' TV` | genre | 0.0856 | 0.071 |
| `duration_minutes` | numeric | 0.0756 | 0.063 |
| `years_to_platform` | numeric | 0.0222 | 0.019 |
| `genre__International Movies` | genre | 0.0222 | 0.019 |
| `n_genres` | numeric | 0.0221 | 0.018 |
| `genre__Dramas` | genre | 0.0177 | 0.015 |
| `release_year` | numeric | 0.0173 | 0.014 |
| `genre__Thrillers` | genre | 0.0168 | 0.014 |
| `genre__Independent Movies` | genre | 0.0162 | 0.014 |
| `genre__Stand-Up Comedy` | genre | 0.0158 | 0.013 |
| `duration_seasons` | numeric | 0.0150 | 0.012 |
| `genre__Crime TV Shows` | genre | 0.0148 | 0.012 |
| `has_director` | numeric | 0.0143 | 0.012 |
| `n_directors` | numeric | 0.0128 | 0.011 |

## Reading the comparison

On Formulation B, most-frequent dummy accuracy is 0.4601 with macro-F1 0.1576. Logistic regression accuracy is 0.6000 and macro-F1 is 0.5647. The gap over the dummy, not the raw accuracy, is the number that matters.

Logistic regression pooled QWK is A=0.5969, B=0.6417, C=0.5792. Ordinal MAE is A=0.5231, B=0.4789, C=0.5377. Adjacent accuracy is A=0.9110, B=0.9287, C=0.9020. Mature→Kids counts are A=117, B=38, C=125.

A's native accuracy can look competitive while still spending capacity on separating 10+ certificates and on an off-ladder `NR` class. C can look accurate because each model has a smaller label set, and it costs a second model plus a router. B is the scale a parent or a moderator can act on.

## Re-run

```
python scripts/design_target.py
```

Data SHA-256: `995a3774af8551d09abd5d2b2a2844a4c0cda28af083176282e42112fb1d200f`
Git SHA: `85ad6e6a7a7fc0d21361f7cb7d37e7dcdbb902ef`
Seed: `42`

The next stage trains, tunes, and calibrates the chosen formulation. It may score the sealed test set. This report does not.
