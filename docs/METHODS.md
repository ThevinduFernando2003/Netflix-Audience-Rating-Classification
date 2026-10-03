# Methods

This note records how the numbers in the reports were produced. It does not add a new experiment. The scripts that emit each report are named below.

## Data contract

`scripts/profile_data.py` writes `reports/data_profile.md` from `data/raw/netflix_titles.csv`. The loader in `src/data/load.py` reads the CSV as strings with `keep_default_na=False`, checks `src/data/schema.py`, and fills a missing `cast` or `description` with an empty string. Empty text is not treated as an observed synopsis.

`src/data/clean.py` rewrites the sentinel `Not Given` to empty, parses `date_added` with an explicit format list (`%m/%d/%Y`, `%Y-%m-%d`, `%Y-%m-%d %H:%M:%S`, `%B %d, %Y`), and splits duration into minutes or seasons. A duration string sitting in `rating` is flagged and excluded. On this file that contamination count is 0, and missing ratings are 0.

## Target design

`scripts/design_target.py` writes `reports/target_design.md`. The sealed split is drawn first: `StratifiedShuffleSplit`, test size 0.15, seed 42, strata equal to the maturity bucket, with NR/UR as their own stratum. The split is stored in `data/processed/sealed_split.json` and reloaded if that file exists, so the test ids do not change. The comparison never scores those 1,319 rows.

Three formulations are fit on the training pool only:

- **A.** Raw certificates. A training-pool class with fewer than 5 rows is folded into `OTHER`. If `OTHER` still has fewer than 5 rows, those rows are dropped. On this pool, `UR` (3) and `NC-17` (1) folded into `OTHER`, and those 4 rows were dropped.
- **B.** The maturity map in `config.yaml`. Kids is `TV-Y`, `TV-G`, `G`. Older Kids is `TV-Y7`, `TV-Y7-FV`, `PG`, `TV-PG`. Teens is `PG-13`, `TV-14`. Mature is `R`, `NC-17`, `TV-MA`. `NR` and `UR` are dropped (69 rows in the training pool), leaving 7,402 labelled rows.
- **C.** One model for movies and one for TV shows, routed by `type`. It is not a film-certificate model plus a TV-certificate model, because 4,430 of 6,126 movies (72.3%) carry a TV certificate.

The selection rule was fixed before the comparison: primary model logistic regression, confirming model random forest, metric pooled out-of-fold quadratic weighted kappa, margin 0.02, tie-break toward B when Mature-as-Kids is not worse. Native accuracy is reported beside a most-frequent dummy and is not the selection metric. Mature is 3,406 of 7,402 Formulation B training rows, so accuracy mostly restates that class. The dummy most-frequent accuracy on B is 0.4601 and its macro-F1 is 0.1576.

## Features

`src/features/engineer.py` builds row-local columns only. Nothing in that module is fit on the data, so it is safe to run before the split. The columns are title, listed genres, type, primary country, primary genre, release year, date parts, years from release to the platform, duration in minutes and in seasons, counts of genres, cast, countries, and directors, title length statistics, a director-present flag, and four lexicon counts (violence, sexual content, substance, family). Description length features stay at 0 because the column is empty.

`src/features/build_pipeline.py` fits the rest inside each fold: title TF-IDF (`strip_accents="unicode"`), a genre multi-label binarizer, one-hot encoding of type, primary country, and primary genre with `min_frequency` 10 and `handle_unknown="ignore"`, and median imputation with an indicator. Standard scaling is the default numeric branch. ComplementNB uses a non-negative branch that clips at zero and does not scale, because standardised features and a negative `years_to_platform` are illegal for that model. HistGradientBoosting receives a dense float32 block through `ToDense`.

Mutual information is computed on the Formulation B training pool only (7,402 rows). Label entropy is 1.1988 nats. The strongest feature is `genre__Children & Family Movies` at 0.1008 nats, 0.084 of the label entropy. No feature reaches 0.90 of that entropy.

## Comparison models

The comparison uses the fixed hyperparameters in `config.yaml` under `comparison.estimators`. They are not a tuned result. The families are most-frequent and stratified dummies, logistic regression, a decision tree, random forest, extra trees, histogram gradient boosting, and ComplementNB. Logistic regression uses the saga solver, `class_weight="balanced"`, `C` 1, `max_iter` 400, and `tol` 0.001. Histogram gradient boosting in this comparison uses `max_iter` 50 and `max_bins` 64. The library default of 255 bins on the dense title matrix was still running after 25 minutes for one formulation, so that budget was cut and written down. The full comparison took 1,543.2 seconds.

Preprocessors are fit on the training side of each fold. Ordinal metrics use levels 0–3. Quadratic weighted kappa uses weights `(i-j)^2 / 9`. Predictions that are not on the ladder (`OTHER`, `NR`, `UR`) are unmapped and are not given a fake level. For Formulation C, quadratic weighted kappa, ordinal MAE, adjacent accuracy, severe-error rate, and the Kids/Mature counts are scored on the concatenated out-of-fold predictions. Macro-F1 and balanced accuracy are the support-weighted means of the movie model and the TV model.

## Tuning

`scripts/tune.py` tunes Formulation B only. `HalvingRandomSearchCV` draws 40 candidates, factor 4, resources from 400 to 2,000 training rows, 5 stratified folds, seed 42, scoring macro-F1, `refit=False`. Families are logistic regression, decision tree, random forest, and histogram gradient boosting. Title TF-IDF `min_df`, `max_features`, and `ngram_range` are searched with the classifier parameters. Histogram gradient boosting stays on `max_iter` in {30, 50} and `max_bins` in {32, 64}. Class weight stays `balanced` or `balanced_subsample`. Inner estimators use `n_jobs=1`.

Each family's halving winner is then scored with a real 5-fold on all 7,402 rows, including training scores. The production model is the family with the highest full-pool macro-F1. Validation curves for that winner use a stratified 2,000-row subset and 3 folds. The winner is refit on every Formulation B training row and written to `models/best_model.joblib`. The sealed test set is not an input.

`LOKY_MAX_CPU_COUNT` is set to 4 before scikit-learn is imported. On this Windows build, joblib's CPU probe calls `wmic` and does not return.

## Holdout evaluation

`scripts/evaluate_test.py` loads the saved forest and the existing split. It scores the 1,306 labelled holdout rows and the 7,402 training rows. The training column is resubstitution: the fitted model predicting the rows it was trained on. It also fits one twin with the same parameters and `class_weight=None`, on the training pool only, and scores that twin once. That comparison does not replace the saved model. Metrics are accuracy, macro-F1, balanced accuracy, quadratic weighted kappa, ordinal MAE, adjacent accuracy, severe-error rate, Mature predicted as Kids, Kids predicted as Mature, a confusion matrix, and top-label expected calibration error in 10 confidence bins.

## Attempts that were not allowed to see the holdout

`scripts/improve_cv.py` builds out-of-fold probabilities for the three tuned models on the training pool. Eight weight mixes were written down before those probabilities were computed. A mix is adopted only when both out-of-fold accuracy and out-of-fold macro-F1 beat the random forest alone.

`scripts/improve_thresholds.py` applies eight pre-registered class-probability scales to those cached probabilities, with the same adoption rule. `scripts/compare_class_weight_oof.py` repeats the forest with `class_weight=None` on the same folds. `scripts/score_improved.py` loads the holdout only if a mix was adopted. None was, so the holdout was not scored a second time for model choice.

A public rating API is not a method in this project. The catalogue is public, so a language model can recite the certificate, which would leak the label.
