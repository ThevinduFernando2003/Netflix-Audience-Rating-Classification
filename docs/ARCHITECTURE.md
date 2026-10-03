# Architecture

The repository is a scikit-learn project driven by `config.yaml`. Seed 42 in that file must match `RANDOM_SEED` in `src/config.py`. A disagreement stops the run. Fitted objects live under `models/` and are gitignored. The sealed split lives under `data/processed/` and is gitignored. Reports under `reports/` are the source of every published number.

## Layout

| path | role |
|---|---|
| `config.yaml` | Seed, paths, sentinels, date formats, bucket map, split size, feature settings, comparison hyperparameters, tuning budget |
| `data/raw/netflix_titles.csv` | Catalogue used by every script |
| `data/lexicons/` | Fixed word lists for the four lexicon counts |
| `data/processed/sealed_split.json` | Train and test `show_id` lists. Created once, then only read |
| `src/data/` | Load, schema, clean, label map, sealed split, training and holdout frames |
| `src/features/` | Row-local features, lexicons, fold-fitted preprocessor |
| `src/models/` | Estimator factory, Formulation B pipelines, search spaces, joblib save |
| `src/evaluation/` | Quadratic weighted kappa and the other ordinal metrics, holdout scoring |
| `src/serving/` | One-title scoring and the HTTP API |
| `app/streamlit_app.py` | Page that calls the same scoring function |
| `scripts/` | Profile, target design, tune, train, one-shot evaluation, training-fold checks |
| `reports/` | Markdown and JSON written by those scripts, plus figures |
| `tests/` | Contract tests on a 102-row fixture and on the prediction response |
| `models/best_model.joblib` | Forest refit on all Formulation B training rows. Local only |

## Training path

```text
CSV
  -> load_raw_catalogue        schema check, empty cast and description
  -> clean_catalogue           sentinels, dates, duration, contamination flags
  -> maturity buckets          config.yaml map; NR/UR stay unlabelled
  -> sealed split              load data/processed/sealed_split.json if it exists
  -> drop test ids
  -> drop unrated
  -> engineer_features         row-local columns only
  -> Pipeline                  preprocessor fit inside each fold, then the classifier
  -> models/best_model.joblib
```

`src/data/dataset.py` exposes two frames. `formulation_b_training_frame` never returns a test id. `formulation_b_holdout_frame` is called only by the one-shot evaluation. Both refuse a frame that mixes the two id sets.

The production pipeline, from `src/models/pipelines.py`, is:

1. `preprocess` — `ColumnTransformer` from `make_preprocessor`. Remainder columns, including the label, are dropped.
2. `clf` — the estimator. Histogram gradient boosting inserts `to_dense` between those steps.

Parameter names in the saved JSON use the prefixes `preprocess__title__tfidf__` and `clf__`. `scripts/train.py` restores `ngram_range` from a JSON list back to a tuple before `set_params`.

## What is fit where

Row-local features (lengths, lexicon counts, primary genre, primary country) are computed before the split and do not use the label. Vocabulary, imputation medians, one-hot levels, and genre columns are learned in `Pipeline.fit`, which cross-validation calls on the training side of each fold only. The final joblib file is that same pipeline fit on all 7,402 labelled training rows.

## Serving path

`src/serving/predict.py` builds one catalogue row, runs `clean_catalogue` and `engineer_features`, and calls `predict` and `predict_proba` on the saved pipeline. The rating field on that row is empty, because the rating is the label. The response lists Kids, Older Kids, Teens, and Mature in that order, even when `classes_` is alphabetical. If the predicted bucket is Kids, the response includes the severe-error warning: a Mature title predicted as Kids is the safety-critical failure.

`src/serving/api.py` is a FastAPI app. `GET /health` reports whether `models/best_model.joblib` exists. `POST /predict` returns 503 when the file is missing and 422 when the title is blank or the type is not `Movie` or `TV Show`. The Streamlit page imports the same function. It does not call the HTTP API, so either process can run alone.

Docker (`Dockerfile`, `docker-compose.yml`) installs `requirements.txt` and `requirements-serve.txt`, copies `config.yaml`, `src/`, `app/`, and `data/lexicons/`, and mounts `./models` read-only. The image does not contain the fitted forest. Port 8000 is the API. Port 8501 is the page.

`.github/workflows/ci.yml` installs the same requirements on Python 3.11 and runs `pytest`. The suite covers cleaning, labels, ordinal metrics, lexicons, the preprocessor, bucket order, the Kids warning, and the health route. It does not need the joblib file.

## Scripts

| script | writes | reads the holdout labels |
|---|---|---|
| `scripts/profile_data.py` | `reports/data_profile.md` | no |
| `scripts/design_target.py` | `reports/target_design.md` | no |
| `scripts/tune.py` | `reports/tuning_results.md`, `reports/best_params.json` | no |
| `scripts/train.py` | `models/best_model.joblib`, `models/metadata.json` | no |
| `scripts/evaluate_test.py` | `reports/test_evaluation.md` | yes, once |
| `scripts/improve_cv.py` | `reports/improvement_cv.md` | no |
| `scripts/improve_thresholds.py` | `reports/threshold_decision.json` | no |
| `scripts/compare_class_weight_oof.py` | `reports/class_weight_oof.json` | no |
| `scripts/score_improved.py` | a note that the vote was not adopted | only if a mix was adopted |

## Configuration that changes behaviour

Edit `config.yaml`, not the Python, to move a certificate between buckets. `maturity_buckets` is the ladder. `unrated_labels` lists certificates that are dropped. `missing_sentinels` must stay quoted, because YAML 1.1 turns bare `n/a`, `NA`, `null`, and `None` into null. `rating_contamination_pattern` must be double-quoted so the backslashes stay regex escapes. `product_formulation` is `B` after the comparison. Tuning limits are under `tuning`.
