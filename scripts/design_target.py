"""Compare label formulations A, B, and C on identical features. Does not score the sealed test set.

The published report is the default run:

    python scripts/design_target.py

`--smoke` is a crash check on a small stratified slice. It does not write the published report.
"""

from __future__ import annotations

import os

# Windows joblib hangs while probing the CPU with wmic. Set this before sklearn imports.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import argparse
import hashlib
import json
import logging
import subprocess
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import RANDOM_SEED, REPO_ROOT, load_config, resolve_path
from src.data.clean import clean_catalogue
from src.data.labels import (
    assert_ratings_covered,
    bucket_table,
    certification_system,
    rating_to_bucket,
    stabilize_raw_labels,
    training_population,
)
from src.data.load import load_raw_catalogue
from src.data.split import save_sealed_split, sealed_split, stratified_folds
from src.evaluation.scoring import native_scores, score_mapped_ordinal
from src.features.build_pipeline import _title_vectorizer, make_preprocessor
from src.features.engineer import NUMERIC_COLUMNS, description_has_text, engineer_features
from src.features.lexicon import load_lexicons
from src.logging_conf import configure_logging
from src.models.estimators import MODEL_NAMES, NONNEGATIVE_MODELS, make_estimator

logger = logging.getLogger(__name__)

# Pre-registered decision rule. Written before the run, applied to the numbers the run produces.
# Primary model is logistic regression so the formulation is what changes.
# Random forest has to agree within QWK_MARGIN or the disagreement is reported and LR still decides.
QWK_MARGIN = 0.02
PRIMARY_MODEL = "logistic_regression"
CONFIRM_MODEL = "random_forest"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _bucket_ordinal_map(config: dict[str, Any]) -> dict[str, int]:
    return {str(bucket["name"]): int(bucket["ordinal"]) for bucket in config["maturity_buckets"]}


def _rating_ordinal_map(config: dict[str, Any]) -> dict[str, int]:
    return {rating: pair[1] for rating, pair in rating_to_bucket(config).items()}


def _prediction_map(formulation: str, config: dict[str, Any]) -> dict[str, int]:
    if formulation == "B":
        return _bucket_ordinal_map(config)
    return _rating_ordinal_map(config)


def _with_truth(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    mapping = rating_to_bucket(config)
    ordinals: list[float] = []
    buckets: list[object] = []
    for rating in frame["rating"].astype(str):
        found = mapping.get(rating)
        if found is None:
            ordinals.append(np.nan)
            buckets.append(pd.NA)
        else:
            ordinals.append(float(found[1]))
            buckets.append(found[0])
    out = frame.copy()
    out["true_ordinal"] = ordinals
    out["true_bucket"] = buckets
    return out


def _strata(frame: pd.DataFrame) -> pd.Series:
    return frame["true_bucket"].map(lambda value: "UNRATED" if pd.isna(value) else str(value))


def _groups(
    frame: pd.DataFrame, formulation: str, config: dict[str, Any], n_splits: int
) -> tuple[list[tuple[str, pd.DataFrame]], dict[str, Any]]:
    min_count = int(config["rare_class_min_count"])
    other = str(config["rare_class_label"])
    info: dict[str, Any] = {}
    if formulation == "B":
        part = frame.loc[frame["true_ordinal"].notna()].copy()
        part["y"] = part["true_bucket"].astype(str)
        info["dropped_unrated"] = int(frame["true_ordinal"].isna().sum())
        return [("all", part.reset_index(drop=True))], info
    if formulation == "A":
        labels, report = stabilize_raw_labels(
            frame["rating"].astype(str), min_count=min_count, n_splits=n_splits, other_label=other
        )
        info["stabilize"] = report
        part = frame.loc[labels.notna()].copy()
        part["y"] = labels.loc[labels.notna()].astype(str).to_numpy()
        return [("all", part.reset_index(drop=True))], info
    groups: list[tuple[str, pd.DataFrame]] = []
    info["groups"] = {}
    unrated = set(config["unrated_labels"])
    for type_name, sub in frame.groupby("type", sort=True):
        part = sub.loc[~sub["rating"].astype(str).isin(unrated)].copy()
        labels, report = stabilize_raw_labels(
            part["rating"].astype(str), min_count=min_count, n_splits=n_splits, other_label=other
        )
        info["groups"][str(type_name)] = {
            "stabilize": report,
            "dropped_unrated": int(sub["rating"].astype(str).isin(unrated).sum()),
            "rows_before": int(len(sub)),
        }
        kept = part.loc[labels.notna()].copy()
        kept["y"] = labels.loc[labels.notna()].astype(str).to_numpy()
        groups.append((str(type_name), kept.reset_index(drop=True)))
    return groups, info


def _densify(matrix: Any) -> np.ndarray:
    if hasattr(matrix, "toarray"):
        return matrix.toarray().astype(np.float32, copy=False)
    return np.asarray(matrix, dtype=np.float32)


def _mean_std(values: list[float]) -> dict[str, float | int | None]:
    array = np.asarray([value for value in values if value is not None and not np.isnan(value)], dtype=float)
    if array.size == 0:
        return {"mean": None, "std": None, "n_folds": 0}
    std = float(array.std(ddof=1)) if array.size > 1 else 0.0
    return {"mean": float(array.mean()), "std": std, "n_folds": int(array.size)}


def _fit_predict_folds(
    frame: pd.DataFrame,
    formulation: str,
    config: dict[str, Any],
    n_splits: int,
) -> dict[str, Any]:
    """Out-of-fold predictions for every comparison model. Preprocessors are fit inside each fold."""
    y = frame["y"].astype(str)
    folds = list(stratified_folds(y, n_splits, RANDOM_SEED).split(frame, y))
    use_description = description_has_text(frame)
    standard_xy: list[tuple[Any, Any, np.ndarray, np.ndarray]] = []
    nonnegative_xy: list[tuple[Any, Any]] = []
    logger.info("Formulation %s: fitting preprocessors on %s folds (%s rows)", formulation, len(folds), len(frame))
    for train_idx, val_idx in folds:
        standard = make_preprocessor(config, use_description=use_description, nonnegative=False)
        standard.fit(frame.iloc[train_idx])
        standard_xy.append(
            (standard.transform(frame.iloc[train_idx]), standard.transform(frame.iloc[val_idx]), train_idx, val_idx)
        )
        nonnegative = make_preprocessor(config, use_description=use_description, nonnegative=True)
        nonnegative.fit(frame.iloc[train_idx])
        nonnegative_xy.append((nonnegative.transform(frame.iloc[train_idx]), nonnegative.transform(frame.iloc[val_idx])))

    pred_map = _prediction_map(formulation, config)
    truth = frame["true_ordinal"].to_numpy(dtype=float)
    model_results: dict[str, Any] = {}
    for name in MODEL_NAMES:
        started = time.perf_counter()
        logger.info("Formulation %s model %s", formulation, name)
        oof = np.empty(len(frame), dtype=object)
        fold_native: dict[str, list[float]] = {key: [] for key in ("accuracy", "balanced_accuracy", "f1_macro")}
        fold_ordinal: dict[str, list[float]] = {
            key: [] for key in ("qwk", "ordinal_mae", "adjacent_accuracy", "severe_error_rate")
        }
        warnings_out: list[str] = []
        for fold_id, (train_idx, val_idx) in enumerate(folds):
            logger.info("Formulation %s model %s fold %s/%s", formulation, name, fold_id + 1, len(folds))
            estimator = make_estimator(name, config, RANDOM_SEED)
            if name in NONNEGATIVE_MODELS:
                x_train, x_val = nonnegative_xy[fold_id]
            else:
                x_train, x_val, _, _ = standard_xy[fold_id]
            if name == "hist_gradient_boosting":
                x_train = _densify(x_train)
                x_val = _densify(x_val)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                estimator.fit(x_train, y.iloc[train_idx])
                predicted = estimator.predict(x_val)
            for item in caught:
                text = str(item.message)
                if "ConvergenceWarning" in item.category.__name__ or "did not converge" in text.lower():
                    warnings_out.append(f"fold {fold_id}: {text}")
            oof[val_idx] = predicted
            native = native_scores(y.iloc[val_idx], predicted)
            for key, value in native.items():
                fold_native[key].append(value)
            ordinal = score_mapped_ordinal(truth[val_idx], predicted, pred_map)
            for key in fold_ordinal:
                value = ordinal.get(key)
                if isinstance(value, float) and not np.isnan(value):
                    fold_ordinal[key].append(float(value))
        pooled_native = native_scores(y, oof)
        pooled_ordinal = score_mapped_ordinal(truth, oof, pred_map)
        model_results[name] = {
            "seconds": round(time.perf_counter() - started, 2),
            "fold_native": {key: _mean_std(values) for key, values in fold_native.items()},
            "fold_ordinal": {key: _mean_std(values) for key, values in fold_ordinal.items()},
            "pooled_native": pooled_native,
            "pooled_ordinal": pooled_ordinal,
            "convergence_warnings": warnings_out[:6],
            "n_convergence_warnings": len(warnings_out),
            "oof_pred": np.asarray(oof, dtype=object),
            "y_true": y.to_numpy(),
            "true_ordinal": truth,
        }
        logger.info(
            "Formulation %s model %s done in %.1fs pooled_qwk=%s macro_f1=%s",
            formulation,
            name,
            model_results[name]["seconds"],
            f"{pooled_ordinal['qwk']:.4f}" if isinstance(pooled_ordinal["qwk"], float) else "nan",
            f"{pooled_native['f1_macro']:.4f}",
        )
    return model_results


def _combine_c(
    group_results: dict[str, dict[str, Any]],
    group_frames: dict[str, pd.DataFrame],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Route each title to its type model and score the concatenated out-of-fold predictions.

    QWK, ordinal MAE, adjacent accuracy, severe-error rate, and the Kids/Mature counts are
    computed on that concatenation, because those metrics share the 4-level ladder.
    Macro-F1 stays the support-weighted mean of the two models' own macro-F1s: concatenating
    the raw certificates would invent one big label set the router never predicts over.
    """
    pred_map = _prediction_map("C", config)
    combined: dict[str, Any] = {}
    for name in MODEL_NAMES:
        pieces = [(len(group_frames[group_name]), group_results[group_name][name]) for group_name in group_results]
        total = sum(size for size, _ in pieces)
        pred = np.concatenate([block["oof_pred"] for _, block in pieces])
        ordinal = np.concatenate([block["true_ordinal"] for _, block in pieces])
        pooled_ordinal = score_mapped_ordinal(ordinal, pred, pred_map)
        pooled_native = {
            "accuracy": float(sum(size * float(block["pooled_native"]["accuracy"]) for size, block in pieces) / total),
            "balanced_accuracy": float(
                sum(size * float(block["pooled_native"]["balanced_accuracy"]) for size, block in pieces) / total
            ),
            "f1_macro": float(sum(size * float(block["pooled_native"]["f1_macro"]) for size, block in pieces) / total),
        }
        fold_ordinal = {}
        fold_native = {}
        for key in ("qwk", "ordinal_mae", "adjacent_accuracy", "severe_error_rate"):
            means = [block["fold_ordinal"][key]["mean"] for _, block in pieces]
            if any(value is None for value in means):
                fold_ordinal[key] = {"mean": None, "std": None, "n_folds": 0}
            else:
                fold_ordinal[key] = {
                    "mean": float(sum(size * float(block["fold_ordinal"][key]["mean"]) for size, block in pieces) / total),
                    "std": None,
                    "n_folds": int(sum(int(block["fold_ordinal"][key]["n_folds"]) for _, block in pieces)),
                }
        for key in ("accuracy", "balanced_accuracy", "f1_macro"):
            fold_native[key] = {
                "mean": float(sum(size * float(block["fold_native"][key]["mean"]) for size, block in pieces) / total),
                "std": None,
                "n_folds": int(sum(int(block["fold_native"][key]["n_folds"]) for _, block in pieces)),
            }
        combined[name] = {
            "seconds": round(sum(block["seconds"] for _, block in pieces), 2),
            "fold_native": fold_native,
            "fold_ordinal": fold_ordinal,
            "pooled_native": pooled_native,
            "pooled_ordinal": pooled_ordinal,
            "convergence_warnings": [],
            "n_convergence_warnings": int(sum(block["n_convergence_warnings"] for _, block in pieces)),
        }
    return combined


def _cell(block: dict[str, Any], family: str, key: str, *, pooled: bool) -> str:
    if pooled:
        source = block["pooled_ordinal" if family == "ordinal" else "pooled_native"]
        value = source[key]
        if isinstance(value, float):
            if np.isnan(value):
                return "n/a"
            return f"{value:.4f}"
        return str(value)
    source = block["fold_ordinal" if family == "ordinal" else "fold_native"][key]
    mean = source["mean"]
    std = source["std"]
    if mean is None:
        return "n/a"
    if std is None:
        return f"{mean:.4f}"
    return f"{mean:.4f} ± {std:.4f}"


def _choose(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Apply the pre-registered rule. Returns the choice and the numbers it used."""

    def qwk(formulation: str, model: str) -> float:
        return float(results[formulation][model]["pooled_ordinal"]["qwk"])

    def mature_as_kids(formulation: str, model: str) -> int:
        return int(results[formulation][model]["pooled_ordinal"]["mature_as_kids"])

    primary = {formulation: qwk(formulation, PRIMARY_MODEL) for formulation in ("A", "B", "C")}
    confirm = {formulation: qwk(formulation, CONFIRM_MODEL) for formulation in ("A", "B", "C")}
    best_primary = max(primary, key=primary.get)
    within = [name for name, value in primary.items() if primary[best_primary] - value <= QWK_MARGIN]
    reasons: list[str] = []
    if best_primary == "B" or "B" in within and primary[best_primary] - primary["B"] <= QWK_MARGIN:
        # Tie-break toward B only when it is inside the margin of the leader.
        leader = best_primary
        if leader != "B" and primary[leader] - primary["B"] <= QWK_MARGIN:
            kids_leader = mature_as_kids(leader, PRIMARY_MODEL)
            kids_b = mature_as_kids("B", PRIMARY_MODEL)
            if kids_b < kids_leader:
                choice = "B"
                reasons.append(
                    f"{PRIMARY_MODEL} pooled QWK puts {leader} first ({primary[leader]:.4f}) and B within "
                    f"{QWK_MARGIN:.2f} ({primary['B']:.4f}). B has fewer Mature→Kids errors "
                    f"({kids_b} vs {kids_leader}), so the tie-break selects B."
                )
            elif kids_b > kids_leader:
                choice = leader
                reasons.append(
                    f"{PRIMARY_MODEL} pooled QWK ties {leader} and B within {QWK_MARGIN:.2f}, "
                    f"and {leader} has fewer Mature→Kids errors ({kids_leader} vs {kids_b})."
                )
            else:
                choice = "B"
                reasons.append(
                    f"{PRIMARY_MODEL} pooled QWK ties {leader} ({primary[leader]:.4f}) and B ({primary['B']:.4f}) "
                    f"within {QWK_MARGIN:.2f}, with the same Mature→Kids count. "
                    "Tie-break selects B: one ordinal scale, no off-ladder class, one model."
                )
        else:
            choice = "B"
            reasons.append(
                f"{PRIMARY_MODEL} pooled QWK ranks B first or alone at the top ({primary['B']:.4f}; "
                f"A={primary['A']:.4f}, C={primary['C']:.4f})."
            )
    else:
        choice = best_primary
        reasons.append(
            f"{PRIMARY_MODEL} pooled QWK selects {choice} ({primary[choice]:.4f}). "
            f"B is {primary[best_primary] - primary['B']:.4f} behind, outside the {QWK_MARGIN:.2f} margin."
        )
    confirm_best = max(confirm, key=confirm.get)
    if confirm_best != choice and abs(confirm[confirm_best] - confirm[choice]) > QWK_MARGIN:
        reasons.append(
            f"{CONFIRM_MODEL} disagrees: its pooled QWK prefers {confirm_best} "
            f"({confirm[confirm_best]:.4f}) over {choice} ({confirm[choice]:.4f}) by more than {QWK_MARGIN:.2f}. "
            "The rule still follows logistic regression and records the disagreement."
        )
    else:
        reasons.append(
            f"{CONFIRM_MODEL} pooled QWK agrees within {QWK_MARGIN:.2f}: "
            + ", ".join(f"{name}={value:.4f}" for name, value in confirm.items())
            + "."
        )
    reasons.append(
        "Native accuracy is not the selection metric. The largest bucket is almost half the rows, "
        "so accuracy mostly restates the majority class. Dummy most-frequent is in the tables for that reason."
    )
    return {"choice": choice, "reasons": reasons, "primary_qwk": primary, "confirm_qwk": confirm}


def _fmt_counts(counts: dict[str, int]) -> str:
    if not counts:
        return "(none)"
    return ", ".join(f"`{label}`={count}" for label, count in sorted(counts.items(), key=lambda item: -item[1]))


def _class_table(frame: pd.DataFrame) -> str:
    counts = frame["y"].astype(str).value_counts()
    lines = ["| label | rows |", "|---|---:|"]
    for label, count in counts.items():
        lines.append(f"| `{label}` | {int(count)} |")
    return "\n".join(lines)


def _metric_matrix(results: dict[str, dict[str, Any]], family: str, key: str, *, pooled: bool) -> str:
    header = "| model | A | B | C |"
    lines = [header, "|---|---:|---:|---:|"]
    for name in MODEL_NAMES:
        cells = [_cell(results[formulation][name], family, key, pooled=pooled) for formulation in ("A", "B", "C")]
        lines.append(f"| `{name}` | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _count_matrix(results: dict[str, dict[str, Any]], key: str) -> str:
    lines = ["| model | A | B | C |", "|---|---:|---:|---:|"]
    for name in MODEL_NAMES:
        cells = [str(int(results[formulation][name]["pooled_ordinal"][key])) for formulation in ("A", "B", "C")]
        lines.append(f"| `{name}` | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _system_counts(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    systems = frame["rating"].astype(str).map(lambda rating: certification_system(rating, config))
    return pd.crosstab(frame["type"], systems)


def _mutual_information(frame: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    """Training-pool only. Asks whether any single feature nearly determines the bucket."""
    y = frame["true_bucket"].astype(str).to_numpy()
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / counts.sum()
    entropy = float(-(probabilities * np.log(probabilities)).sum())
    numeric = frame[NUMERIC_COLUMNS].apply(pd.to_numeric, errors="coerce")
    imputer = SimpleImputer(strategy="median")
    numeric_filled = imputer.fit_transform(numeric)
    numeric_mi = mutual_info_classif(numeric_filled, y, discrete_features=False, random_state=RANDOM_SEED)
    numeric_rows = [
        {"feature": column, "mi": float(score), "kind": "numeric"}
        for column, score in zip(NUMERIC_COLUMNS, numeric_mi, strict=True)
    ]

    from src.features.build_pipeline import GenreBinarizer

    genres = GenreBinarizer().fit(frame[["listed_in"]])
    genre_matrix = genres.transform(frame[["listed_in"]])
    genre_mi = mutual_info_classif(genre_matrix, y, discrete_features=True, random_state=RANDOM_SEED)
    genre_rows = [
        {"feature": name, "mi": float(score), "kind": "genre"}
        for name, score in zip(genres.get_feature_names_out(), genre_mi, strict=True)
    ]

    title_vectorizer: TfidfVectorizer = _title_vectorizer(config)
    title_matrix = title_vectorizer.fit_transform(frame["title"].astype(str))
    document_frequency = np.asarray((title_matrix > 0).sum(axis=0)).ravel()
    kept = np.flatnonzero(document_frequency >= 5)
    if kept.size > 1500:
        kept = kept[np.argsort(document_frequency[kept])[::-1][:1500]]
    present = (title_matrix[:, kept] > 0).astype(np.int8)
    title_mi_parts: list[np.ndarray] = []
    width = present.shape[1]
    for start in range(0, width, 500):
        chunk = present[:, start : start + 500].toarray()
        title_mi_parts.append(mutual_info_classif(chunk, y, discrete_features=True, random_state=RANDOM_SEED))
    title_mi = np.concatenate(title_mi_parts) if title_mi_parts else np.asarray([])
    title_names = title_vectorizer.get_feature_names_out()[kept]
    top_title_index = np.argsort(title_mi)[::-1][:15]
    title_rows = [
        {"feature": f"title__{title_names[index]}", "mi": float(title_mi[index]), "kind": "title_token"}
        for index in top_title_index
    ]
    ranked = sorted(numeric_rows + genre_rows + title_rows, key=lambda item: item["mi"], reverse=True)
    best = ranked[0]
    best["ratio_to_label_entropy"] = float(best["mi"] / entropy) if entropy else None
    flagged = [
        {**item, "ratio_to_label_entropy": float(item["mi"] / entropy)}
        for item in ranked
        if entropy and item["mi"] / entropy >= 0.9
    ]
    return {
        "label_entropy_nats": entropy,
        "n_rows": int(len(frame)),
        "best": best,
        "near_perfect": flagged,
        "top": [{**item, "ratio_to_label_entropy": float(item["mi"] / entropy) if entropy else None} for item in ranked[:15]],
        "description_vectorizer_fitted": False,
        "description_note": (
            "No description TF-IDF was fitted. The description column is empty on every row of this extract, "
            "so a vectorizer would have an empty vocabulary. Title-token MI uses tokens with document "
            f"frequency at least 5, capped at the {int(kept.size)} highest-frequency of those tokens."
        ),
        "lexicon_sizes": {name: len(terms) for name, terms in load_lexicons(config["lexicons"], REPO_ROOT).items()},
    }


def _plot_counts(raw_counts: pd.Series, bucket_counts: pd.Series, bucket_order: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 6), constrained_layout=True)
    raw_order = raw_counts.sort_values(ascending=True)
    axes[0].barh(raw_order.index.astype(str), raw_order.values, color="#4C78A8")
    axes[0].set_xlabel("Titles")
    axes[0].set_title("Raw rating")
    palette = {"Kids": "#2E7D32", "Older Kids": "#F9A825", "Teens": "#EF6C00", "Mature": "#C62828"}
    ordered = bucket_counts.reindex(bucket_order)
    axes[1].barh(
        [str(label) for label in ordered.index[::-1]],
        ordered.values[::-1],
        color=[palette.get(str(label), "#4C78A8") for label in ordered.index[::-1]],
    )
    axes[1].set_xlabel("Titles")
    axes[1].set_title("Maturity bucket")
    fig.suptitle("Label space before and after the maturity ladder")
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _placement_section() -> str:
    return "\n".join(
        [
            "## Where each certificate sits, and why",
            "",
            "The ladder used as a guide is "
            "`TV-Y < TV-G < TV-Y7 < TV-PG < PG < TV-14 < PG-13 < R < TV-MA < NC-17`. "
            "`G` is the film certificate next to `TV-G`. Buckets follow that order. "
            "They do not follow alphabetical order.",
            "",
            "- **Kids (0)** — `TV-Y`, `TV-G`, `G`. All three are unrestricted: young-child TV, general TV, and general film. A parent does not need a warning.",
            "- **Older Kids (1)** — `TV-Y7`, `TV-Y7-FV`, `PG`, `TV-PG`. `TV-Y7` is aimed at children over 7. `TV-Y7-FV` is that same certificate with a fantasy-violence flag (6 titles), not a separate maturity level. `TV-PG` and `PG` are the judgement call: the ladder puts `TV-PG` just below `PG`, and both mean parental guidance rather than a teen restriction. They sit next to each other, below `TV-14` / `PG-13`, so they share a bucket. Splitting them would rebuild the two-certificate problem inside one boundary.",
            "- **Teens (2)** — `PG-13`, `TV-14`. Both are the 'not for younger children' band. The ladder puts `TV-14` just below `PG-13`. They are one product decision apart, not a different audience.",
            "- **Mature (3)** — `R`, `NC-17`, `TV-MA`. `R` and `TV-MA` are the adult certificates. `NC-17` is stricter than `R` and has 3 titles, which cannot support its own class under 5-fold stratification. It stays at the top of the ladder instead of being dropped or called a teen rating.",
            "- **`NR` and `UR` are not on the ladder.** They mean the title was not certified. Formulation B drops them. The count is in the exclusions section.",
            "",
        ]
    )


def render_report(payload: dict[str, Any]) -> str:
    choice = payload["choice"]["choice"]
    names = {"A": "raw multi-class", "B": "maturity buckets", "C": "per-type models"}
    lines: list[str] = []
    lines.append("# Target design")
    lines.append("")
    lines.append(f"## Product choice: Formulation {choice} — {names[choice]}")
    lines.append("")
    for reason in payload["choice"]["reasons"]:
        lines.append(f"- {reason}")
    lines.append("")
    lines.append(
        f"Primary comparison model: `{PRIMARY_MODEL}`. Confirming model: `{CONFIRM_MODEL}`. "
        f"Tie margin: {QWK_MARGIN:.2f} pooled out-of-fold quadratic weighted kappa (QWK). "
        "The sealed 15% test set was not scored. Figures below are cross-validated on the training pool only."
    )
    lines.append("")
    lines.append("Pooled QWK used by the rule:")
    lines.append("")
    lines.append("| model | A | B | C |")
    lines.append("|---|---:|---:|---:|")
    for model, store in (("primary_qwk", PRIMARY_MODEL), ("confirm_qwk", CONFIRM_MODEL)):
        values = payload["choice"][model]
        lines.append(
            f"| `{store}` | {values['A']:.4f} | {values['B']:.4f} | {values['C']:.4f} |"
        )
    lines.append("")
    lines.append("## What was compared")
    lines.append("")
    lines.append(
        "Formulation A keeps raw certificates and folds any training-pool class with fewer than "
        f"{payload['rare_class_min_count']} rows into `{payload['rare_class_label']}`. "
        "If that leftover class still cannot support 5-fold stratification, those rows are dropped and the count is below."
    )
    lines.append("")
    lines.append(
        "Formulation B maps both certification systems onto one 4-level ordinal scale. "
        "The map lives in `config.yaml` under `maturity_buckets`."
    )
    lines.append("")
    lines.append(
        "Formulation C fits one model for movies and one model for TV shows, routed by `type`. "
        "It is not a clean film-certificate model plus a clean TV-certificate model. "
        "On this file, most movies are labelled with a TV certificate, so a film-only movie model would discard them."
    )
    lines.append("")
    lines.append("### Certification system by type")
    lines.append("")
    lines.append("Counts are the full catalogue, including the sealed test rows. This table is a description of the labels, not a model score.")
    lines.append("")
    system = payload["system_counts"]
    header = [str(column) for column in system["columns"]]
    lines.append("| " + " | ".join(header) + " |")
    aligns = ["---"] + ["---:"] * (len(header) - 1)
    lines.append("|" + "|".join(aligns) + "|")
    for row in system["rows"]:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    lines.append("")
    lines.append(payload["system_prose"])
    lines.append("")
    lines.append(_placement_section())
    lines.append("## Rows excluded from a formulation")
    lines.append("")
    lines.append(f"- Contaminated `rating` (duration string): **{payload['n_contaminated']}**. Excluded from every formulation.")
    lines.append(f"- Missing `rating`: **{payload['n_missing_rating']}**. Excluded from every formulation. Kept as the unlabelled demo pool.")
    lines.append(f"- Sealed test rows, not scored: **{payload['n_test']}**.")
    lines.append(f"- Training-pool rows available to the comparison: **{payload['n_train_pool']}**.")
    lines.append(f"- Formulation B drops unrated (`NR`, `UR`) rows inside the training pool: **{payload['formulations']['B']['info'].get('dropped_unrated', 0)}**.")
    lines.append("")
    lines.append("### Formulation A label stabilisation (training pool)")
    lines.append("")
    stab = payload["formulations"]["A"]["info"]["stabilize"]
    lines.append(f"- Folded into `{payload['rare_class_label']}`: {_fmt_counts(stab['folded_into_other'])}.")
    lines.append(f"- `{payload['rare_class_label']}` rows dropped because the folded class still had fewer than {payload['n_splits']} rows: **{stab['dropped_other']}**.")
    lines.append(f"- Other classes dropped for the same reason: {_fmt_counts(stab['dropped_still_small'])}.")
    lines.append("")
    lines.append("### Formulation C label stabilisation (training pool, per type)")
    lines.append("")
    for type_name, block in payload["formulations"]["C"]["info"]["groups"].items():
        report = block["stabilize"]
        lines.append(f"**{type_name}** — rows before the unrated filter: {block['rows_before']}; unrated dropped: {block['dropped_unrated']}.")
        lines.append("")
        lines.append(f"- Folded into `{payload['rare_class_label']}`: {_fmt_counts(report['folded_into_other'])}.")
        lines.append(f"- Folded class dropped: **{report['dropped_other']}**.")
        lines.append(f"- Still too small and dropped: {_fmt_counts(report['dropped_still_small'])}.")
        lines.append("")
    lines.append("## Class counts in the training pool after stabilisation")
    lines.append("")
    for formulation in ("A", "B"):
        lines.append(f"### Formulation {formulation}")
        lines.append("")
        lines.append(payload["formulations"][formulation]["class_tables"]["all"])
        lines.append("")
    lines.append("### Formulation C")
    lines.append("")
    for type_name, table in payload["formulations"]["C"]["class_tables"].items():
        lines.append(f"**{type_name}**")
        lines.append("")
        lines.append(table)
        lines.append("")
    lines.append("## Protocol")
    lines.append("")
    lines.append(
        f"Seed `{payload['seed']}`. Stratified holdout `{payload['test_size']:.0%}` using the bucket, "
        "with unrated rows as their own stratum so the split does not depend on a 3-row class. "
        f"Model comparison uses {payload['n_splits']}-fold `StratifiedKFold(shuffle=True)` on each formulation's own training labels. "
        "Formulation C runs that split separately inside movies and inside TV shows."
    )
    lines.append("")
    lines.append(
        "Every vectorizer, imputer, scaler, and one-hot encoder is fit on the training side of the fold only. "
        "Title TF-IDF, genre indicators, type, primary country, primary genre, numeric metadata, and the lexicon counts are shared. "
        "Description TF-IDF is omitted because that column is empty. "
        "`ComplementNB` uses a non-negative branch (median impute, no standardisation, clipped at zero) because standardisation produces negatives and `years_to_platform` can be negative."
    )
    lines.append("")
    lines.append(
        "Hyperparameters are the fixed comparison settings in `config.yaml` under `comparison.estimators`. "
        "They are not tuned. Tuning the winner is the next stage. "
        f"Total comparison wall clock: **{payload['wall_clock_seconds']:.1f} seconds**."
    )
    lines.append("")
    lines.append(
        "Ordinal metrics map predictions back onto the 4-level ladder. "
        "`OTHER`, `NR`, and `UR` predictions are not given a fake level: they count as unmapped. "
        "QWK uses weights `(i-j)^2 / 9` on levels 0–3. "
        "`mature_as_kids` is the safety-critical error: a Mature title predicted as Kids."
    )
    lines.append("")
    lines.append(
        "For Formulation C, QWK, ordinal MAE, adjacent accuracy, severe-error rate, and the Kids/Mature counts "
        "are scored on the concatenated out-of-fold predictions, on the shared 4-level ladder. "
        "Macro-F1 and balanced accuracy are the support-weighted means of the movie model and the TV model: "
        "those metrics are not meaningful on the union of two different certificate sets. "
        "Routed accuracy equals that same weighted mean. Per-type mean ± std follows."
    )
    lines.append("")
    lines.append(
        "HistGradientBoosting uses `max_iter=50` and `max_bins=64`. "
        "The library default of 255 bins on the dense title matrix was still running after 25 minutes for one formulation, "
        "so the comparison budget was reduced and written down in `config.yaml` and `docs/DECISIONS.md`. "
        "These are not tuned parameters."
    )
    lines.append("")
    lines.append("## Pooled out-of-fold results")
    lines.append("")
    lines.append("### QWK")
    lines.append("")
    lines.append(_metric_matrix(payload["model_results"], "ordinal", "qwk", pooled=True))
    lines.append("")
    lines.append("### Macro-F1 (native label space; C is the support-weighted mean of the two models)")
    lines.append("")
    lines.append(_metric_matrix(payload["model_results"], "native", "f1_macro", pooled=True))
    lines.append("")
    lines.append("### Accuracy")
    lines.append("")
    lines.append(_metric_matrix(payload["model_results"], "native", "accuracy", pooled=True))
    lines.append("")
    lines.append("### Balanced accuracy")
    lines.append("")
    lines.append(_metric_matrix(payload["model_results"], "native", "balanced_accuracy", pooled=True))
    lines.append("")
    lines.append("### Ordinal MAE")
    lines.append("")
    lines.append(_metric_matrix(payload["model_results"], "ordinal", "ordinal_mae", pooled=True))
    lines.append("")
    lines.append("### Adjacent accuracy (within 1 level)")
    lines.append("")
    lines.append(_metric_matrix(payload["model_results"], "ordinal", "adjacent_accuracy", pooled=True))
    lines.append("")
    lines.append("### Severe-error rate (off by 2 or more levels, mapped predictions only)")
    lines.append("")
    lines.append(_metric_matrix(payload["model_results"], "ordinal", "severe_error_rate", pooled=True))
    lines.append("")
    lines.append("### Unmapped prediction rate")
    lines.append("")
    lines.append(_metric_matrix(payload["model_results"], "ordinal", "unmapped_prediction_rate", pooled=True))
    lines.append("")
    lines.append("### Mature predicted as Kids (safety-critical count)")
    lines.append("")
    lines.append(_count_matrix(payload["model_results"], "mature_as_kids"))
    lines.append("")
    lines.append("### Kids predicted as Mature (count)")
    lines.append("")
    lines.append(_count_matrix(payload["model_results"], "kids_as_mature"))
    lines.append("")
    lines.append("## 5-fold mean ± std")
    lines.append("")
    lines.append("### Formulation A — QWK")
    lines.append("")
    lines.append(_one_formulation_fold_table(payload["model_results"]["A"], "ordinal", "qwk"))
    lines.append("")
    lines.append("### Formulation A — macro-F1")
    lines.append("")
    lines.append(_one_formulation_fold_table(payload["model_results"]["A"], "native", "f1_macro"))
    lines.append("")
    lines.append("### Formulation B — QWK")
    lines.append("")
    lines.append(_one_formulation_fold_table(payload["model_results"]["B"], "ordinal", "qwk"))
    lines.append("")
    lines.append("### Formulation B — macro-F1")
    lines.append("")
    lines.append(_one_formulation_fold_table(payload["model_results"]["B"], "native", "f1_macro"))
    lines.append("")
    lines.append("### Formulation B — balanced accuracy")
    lines.append("")
    lines.append(_one_formulation_fold_table(payload["model_results"]["B"], "native", "balanced_accuracy"))
    lines.append("")
    lines.append("### Formulation C per type — QWK mean ± std")
    lines.append("")
    for type_name, results in payload["formulations"]["C"]["per_type_results"].items():
        lines.append(f"**{type_name} — QWK**")
        lines.append("")
        lines.append(_one_formulation_fold_table(results, "ordinal", "qwk"))
        lines.append("")
        lines.append(f"**{type_name} — macro-F1**")
        lines.append("")
        lines.append(_one_formulation_fold_table(results, "native", "f1_macro"))
        lines.append("")
    lines.append("## Mutual information")
    lines.append("")
    mi = payload["mutual_information"]
    best = mi["best"]
    lines.append(
        f"Computed on the Formulation B training pool only ({mi['n_rows']} rows). "
        f"Label entropy is {mi['label_entropy_nats']:.4f} nats. "
        f"The strongest single feature is `{best['feature']}` ({best['kind']}) "
        f"with MI {best['mi']:.4f} nats, {best['ratio_to_label_entropy']:.3f} of the label entropy."
    )
    lines.append("")
    if mi["near_perfect"]:
        lines.append("Features at or above 0.90 of label entropy, which would mean near-perfect separation:")
        lines.append("")
        for item in mi["near_perfect"]:
            lines.append(f"- `{item['feature']}` MI={item['mi']:.4f} ratio={item['ratio_to_label_entropy']:.3f}")
    else:
        lines.append(
            "No numeric feature, genre indicator, or title token reaches 0.90 of the label entropy. "
            "Nothing in the feature set restates the certificate."
        )
    lines.append("")
    lines.append(mi["description_note"])
    lines.append("")
    lines.append("| feature | kind | MI (nats) | share of label entropy |")
    lines.append("|---|---|---:|---:|")
    for item in mi["top"]:
        lines.append(
            f"| `{item['feature']}` | {item['kind']} | {item['mi']:.4f} | {item['ratio_to_label_entropy']:.3f} |"
        )
    lines.append("")
    lines.append("## Reading the comparison")
    lines.append("")
    lines.append(payload["reading"])
    lines.append("")
    lines.append("## Re-run")
    lines.append("")
    lines.append("```")
    lines.append("python scripts/design_target.py")
    lines.append("```")
    lines.append("")
    lines.append(f"Data SHA-256: `{payload['data_sha256']}`")
    lines.append(f"Git SHA: `{payload['git_sha']}`")
    lines.append(f"Seed: `{payload['seed']}`")
    lines.append("")
    lines.append("The next stage trains, tunes, and calibrates the chosen formulation. It may score the sealed test set. This report does not.")
    lines.append("")
    return "\n".join(lines)


def _one_formulation_fold_table(results: dict[str, Any], family: str, key: str) -> str:
    lines = ["| model | mean ± std | folds |", "|---|---:|---:|"]
    for name in MODEL_NAMES:
        source = results[name]["fold_ordinal" if family == "ordinal" else "fold_native"][key]
        if source["mean"] is None:
            shown = "n/a"
        elif source["std"] is None:
            shown = f"{source['mean']:.4f}"
        else:
            shown = f"{source['mean']:.4f} ± {source['std']:.4f}"
        lines.append(f"| `{name}` | {shown} | {source['n_folds']} |")
    return "\n".join(lines)


def _reading(results: dict[str, dict[str, Any]]) -> str:
    b_lr = results["B"][PRIMARY_MODEL]["pooled_ordinal"]
    a_lr = results["A"][PRIMARY_MODEL]["pooled_ordinal"]
    c_lr = results["C"][PRIMARY_MODEL]["pooled_ordinal"]
    b_dummy = results["B"]["dummy_most_frequent"]["pooled_native"]
    b_lr_native = results["B"][PRIMARY_MODEL]["pooled_native"]
    return "\n".join(
        [
            f"On Formulation B, most-frequent dummy accuracy is {b_dummy['accuracy']:.4f} "
            f"with macro-F1 {b_dummy['f1_macro']:.4f}. "
            f"Logistic regression accuracy is {b_lr_native['accuracy']:.4f} and macro-F1 is {b_lr_native['f1_macro']:.4f}. "
            "The gap over the dummy, not the raw accuracy, is the number that matters.",
            "",
            f"Logistic regression pooled QWK is A={a_lr['qwk']:.4f}, B={b_lr['qwk']:.4f}, C={c_lr['qwk']:.4f}. "
            f"Ordinal MAE is A={a_lr['ordinal_mae']:.4f}, B={b_lr['ordinal_mae']:.4f}, C={c_lr['ordinal_mae']:.4f}. "
            f"Adjacent accuracy is A={a_lr['adjacent_accuracy']:.4f}, B={b_lr['adjacent_accuracy']:.4f}, C={c_lr['adjacent_accuracy']:.4f}. "
            f"Mature→Kids counts are A={int(a_lr['mature_as_kids'])}, B={int(b_lr['mature_as_kids'])}, C={int(c_lr['mature_as_kids'])}.",
            "",
            "A's native accuracy can look competitive while still spending capacity on separating 10+ certificates "
            "and on an off-ladder `NR` class. C can look accurate because each model has a smaller label set, "
            "and it costs a second model plus a router. B is the scale a parent or a moderator can act on.",
        ]
    )


def _maybe_sample(frame: pd.DataFrame, n_rows: int) -> pd.DataFrame:
    if len(frame) <= n_rows:
        return frame
    strata = _strata(frame)
    # A stratum smaller than 2 cannot be stratified; fold it into the largest stratum for the sample only.
    counts = strata.value_counts()
    safe = strata.where(~strata.isin(counts[counts < 2].index), counts.idxmax())
    splitter_frame = frame
    from sklearn.model_selection import StratifiedShuffleSplit

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=n_rows, random_state=RANDOM_SEED)
    _, keep = next(splitter.split(splitter_frame, safe))
    return frame.iloc[keep].reset_index(drop=True)


def _strip_predictions(results: dict[str, Any]) -> None:
    """Drop out-of-fold arrays before the report payload is serialized."""
    for block in results.values():
        for key in ("oof_pred", "y_true", "true_ordinal"):
            block.pop(key, None)


def run(smoke: bool) -> None:
    configure_logging()
    started = time.perf_counter()
    config = load_config()
    n_splits = 2 if smoke else int(config["split"]["n_splits"])
    if smoke:
        for name in ("random_forest", "extra_trees"):
            config["comparison"]["estimators"][name]["n_estimators"] = 15
        config["comparison"]["estimators"]["hist_gradient_boosting"]["max_iter"] = 20
        config["comparison"]["estimators"]["logistic_regression"]["max_iter"] = 80
    raw_path = resolve_path(config, "raw_csv")
    raw = load_raw_catalogue(raw_path, config)
    assert_ratings_covered(raw["rating"].tolist(), config)
    cleaned = clean_catalogue(raw, config)
    population = _with_truth(training_population(cleaned), config)
    strata = _strata(population)
    split_payload = sealed_split(
        population["show_id"],
        strata,
        test_size=float(config["split"]["test_size"]),
        seed=RANDOM_SEED,
    )
    split_payload["seed"] = RANDOM_SEED
    split_payload["test_size"] = float(config["split"]["test_size"])
    split_payload["stratify"] = "maturity bucket, with NR/UR as UNRATED"
    split_payload["data_sha256"] = _sha256(raw_path)
    if not smoke:
        save_sealed_split(split_payload, resolve_path(config, "sealed_split"))
    test_ids = set(split_payload["test_show_ids"])
    train = population.loc[~population["show_id"].isin(test_ids)].copy()
    if smoke:
        train = _maybe_sample(train, 800)
    engineered = engineer_features(train, config)
    # engineer_features drops helper columns. Put the label columns back via show_id.
    labels = train[["show_id", "rating", "true_ordinal", "true_bucket"]]
    modelled = engineered.merge(labels, on="show_id", how="left", validate="one_to_one")
    if set(modelled["show_id"]).intersection(test_ids):
        raise RuntimeError("Sealed test show_ids entered the comparison frame.")

    formulations: dict[str, Any] = {}
    model_results: dict[str, Any] = {}
    for formulation in ("A", "B", "C"):
        groups, info = _groups(modelled, formulation, config, n_splits)
        per_type_results: dict[str, Any] = {}
        frames: dict[str, pd.DataFrame] = {}
        class_tables: dict[str, str] = {}
        for group_name, group_frame in groups:
            logger.info("Running formulation %s group %s (%s rows)", formulation, group_name, len(group_frame))
            per_type_results[group_name] = _fit_predict_folds(group_frame, formulation, config, n_splits)
            frames[group_name] = group_frame
            class_tables[group_name] = _class_table(group_frame)
        if formulation == "C":
            model_results[formulation] = _combine_c(per_type_results, frames, config)
        else:
            model_results[formulation] = per_type_results["all"]
        formulations[formulation] = {
            "info": info,
            "class_tables": class_tables,
            "per_type_results": per_type_results if formulation == "C" else {},
            "n_rows": {name: int(len(frame)) for name, frame in frames.items()},
        }

    for block in formulations.values():
        for results in block["per_type_results"].values():
            _strip_predictions(results)
    for results in model_results.values():
        _strip_predictions(results)

    bucket_rows = modelled.loc[modelled["true_ordinal"].notna()].copy()
    mutual = _mutual_information(bucket_rows, config)
    choice = _choose(model_results)
    system = _system_counts(population, config)
    system_rows = [[index, *[int(system.loc[index, column]) for column in system.columns]] for index in system.index]
    movie_tv = int(system.loc["Movie", "tv"]) if "tv" in system.columns else 0
    movie_film = int(system.loc["Movie", "film"]) if "film" in system.columns else 0
    movie_n = int(system.loc["Movie"].sum())
    system_prose = (
        f"Movies: {movie_n} titles, of which {movie_tv} ({movie_tv / movie_n:.1%}) carry a TV certificate "
        f"and {movie_film} ({movie_film / movie_n:.1%}) carry a film certificate. "
        "A movie model restricted to film certificates would ignore the majority of movies. "
        "That is why Formulation C is routed by `type` and trained on the certificates each type actually uses."
    )
    raw_counts = population["rating"].astype(str).value_counts()
    bucket_counts = population["true_bucket"].dropna().astype(str).value_counts()
    bucket_order = [str(bucket["name"]) for bucket in config["maturity_buckets"]]
    figure_path = REPO_ROOT / "reports" / "figures" / "label_counts_raw_vs_buckets.png"
    if not smoke:
        _plot_counts(raw_counts, bucket_counts, bucket_order, figure_path)

    payload: dict[str, Any] = {
        "choice": choice,
        "seed": RANDOM_SEED,
        "test_size": float(config["split"]["test_size"]),
        "n_splits": n_splits,
        "rare_class_min_count": int(config["rare_class_min_count"]),
        "rare_class_label": str(config["rare_class_label"]),
        "n_contaminated": int(cleaned["rating_contaminated"].sum()),
        "n_missing_rating": int(cleaned["rating_missing"].sum()),
        "n_test": len(split_payload["test_show_ids"]),
        "n_train_pool": int(len(train)) if not smoke else int(len(modelled)),
        "formulations": formulations,
        "model_results": model_results,
        "mutual_information": mutual,
        "system_counts": {"columns": ["type", *list(system.columns)], "rows": system_rows},
        "system_prose": system_prose,
        "wall_clock_seconds": time.perf_counter() - started,
        "data_sha256": split_payload["data_sha256"],
        "git_sha": _git_sha(),
        "reading": _reading(model_results),
        "bucket_table": bucket_table(config).to_dict(orient="records"),
        "smoke": smoke,
    }
    # JSON cannot store the reading twice as a problem; model results contain numpy types already cast.
    report_dir = REPO_ROOT / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    if smoke:
        out_md = report_dir / "_smoke_target_design.md"
        out_json = report_dir / "_smoke_target_design.json"
    else:
        out_md = report_dir / "target_design.md"
        out_json = report_dir / "target_design_metrics.json"
        _set_product_formulation(choice["choice"])
    out_md.write_text(render_report(payload), encoding="utf-8")
    out_json.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
    logger.info("Wrote %s", out_md)
    logger.info("Product formulation: %s", choice["choice"])


def _set_product_formulation(choice: str) -> None:
    path = REPO_ROOT / "config.yaml"
    text = path.read_text(encoding="utf-8")
    replaced, count = __import__("re").subn(
        r"(?m)^product_formulation:\s*\S+\s*$",
        f"product_formulation: {choice}",
        text,
        count=1,
    )
    if count != 1:
        raise RuntimeError("Could not update product_formulation in config.yaml.")
    path.write_text(replaced, encoding="utf-8")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return None
    if isinstance(value, (np.floating,)):
        number = float(value)
        return None if np.isnan(number) or np.isinf(number) else number
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare rating label formulations.")
    parser.add_argument("--smoke", action="store_true", help="Small crash check. Does not write the published report.")
    args = parser.parse_args()
    run(smoke=args.smoke)


if __name__ == "__main__":
    main()
