"""Tune Formulation B. The sealed test set is not scored.

HalvingRandomSearchCV draws 40 candidates. Early rungs see at most `max_resources`
training rows so the search finishes on a laptop. Each family's winner is then
scored with a full 5-fold on every training row. That full-fold number is the one
in the report.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.model_selection import HalvingRandomSearchCV, StratifiedKFold, validation_curve
from sklearn.model_selection import cross_validate

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
os.environ["PYTHONPATH"] = str(_ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")

from scripts.train import fit_and_save
from src.config import RANDOM_SEED, REPO_ROOT, load_config
from src.data.dataset import formulation_b_training_frame
from src.data.split import stratified_folds
from src.evaluation.ordinal import quadratic_weighted_kappa
from src.logging_conf import configure_logging
from src.models.pipelines import BUCKET_ORDER, make_model_pipeline, search_spaces

logger = logging.getLogger(__name__)

ORDINAL = {name: index for index, name in enumerate(BUCKET_ORDER)}


def _qwk_scorer(estimator: Any, x: Any, y: Any) -> float:
    predicted = estimator.predict(x)
    truth = [ORDINAL[str(label)] for label in y]
    guess = [ORDINAL[str(label)] for label in predicted]
    return float(quadratic_weighted_kappa(truth, guess, labels=[0, 1, 2, 3]))


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return None if np.isnan(number) else number
    if isinstance(value, (np.integer,)):
        return int(value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    return str(value)


def _search_one(
    name: str,
    frame: Any,
    y: Any,
    config: dict[str, Any],
    tuning: dict[str, Any],
) -> dict[str, Any]:
    pipeline = make_model_pipeline(name, config, use_description=False, seed=RANDOM_SEED)
    space = search_spaces()[name]
    cv = StratifiedKFold(n_splits=int(tuning["cv_splits"]), shuffle=True, random_state=RANDOM_SEED)
    search = HalvingRandomSearchCV(
        pipeline,
        space,
        n_candidates=int(tuning["n_candidates"]),
        factor=int(tuning["factor"]),
        min_resources=int(tuning["min_resources"]),
        max_resources=int(tuning["max_resources"]),
        cv=cv,
        scoring="f1_macro",
        refit=False,
        error_score=np.nan,
        random_state=RANDOM_SEED,
        n_jobs=int(tuning["n_jobs"]),
        verbose=0,
    )
    started = time.perf_counter()
    logger.info("Searching %s (%s candidates, max_resources=%s)", name, tuning["n_candidates"], tuning["max_resources"])
    search.fit(frame, y)
    elapsed = time.perf_counter() - started
    best_index = int(search.best_index_)
    results = search.cv_results_
    summary = {
        "model": name,
        "best_params": _jsonable(search.best_params_),
        "halving_best_f1_macro": float(search.best_score_),
        "halving_best_f1_macro_std": float(results["std_test_score"][best_index]),
        "n_candidates_drawn": int(tuning["n_candidates"]),
        "seconds": round(elapsed, 1),
        "rungs": _rung_table(results),
    }
    logger.info(
        "%s halving f1_macro %.4f ± %.4f in %.0fs",
        name,
        summary["halving_best_f1_macro"],
        summary["halving_best_f1_macro_std"],
        elapsed,
    )
    return summary


def _rung_table(results: dict[str, Any]) -> list[dict[str, Any]]:
    iterations = results.get("iter")
    if iterations is None:
        return []
    rows: list[dict[str, Any]] = []
    for rung in sorted(set(int(item) for item in iterations)):
        mask = [int(item) == rung for item in iterations]
        scores = [float(score) for score, keep in zip(results["mean_test_score"], mask, strict=True) if keep and score == score]
        resources = [int(value) for value, keep in zip(results["n_resources"], mask, strict=True) if keep]
        rows.append(
            {
                "rung": rung,
                "candidates": int(sum(mask)),
                "resources": resources[0] if resources else None,
                "best_f1_macro": None if not scores else max(scores),
            }
        )
    return rows


def _full_cv(name: str, params: dict[str, Any], frame: Any, y: Any, config: dict[str, Any], n_splits: int) -> dict[str, Any]:
    pipeline = make_model_pipeline(name, config, use_description=False, seed=RANDOM_SEED)
    from scripts.train import apply_params

    apply_params(pipeline, params)
    cv = stratified_folds(y, n_splits, RANDOM_SEED)
    logger.info("Full %s-fold score for %s", n_splits, name)
    started = time.perf_counter()
    scores = cross_validate(
        pipeline,
        frame,
        y,
        cv=cv,
        scoring={
            "f1_macro": "f1_macro",
            "balanced_accuracy": "balanced_accuracy",
            "qwk": _qwk_scorer,
        },
        n_jobs=1,
        return_train_score=True,
    )
    elapsed = time.perf_counter() - started
    summary: dict[str, Any] = {"seconds": round(elapsed, 1)}
    for key in ("f1_macro", "balanced_accuracy", "qwk"):
        test = np.asarray(scores[f"test_{key}"], dtype=float)
        train = np.asarray(scores[f"train_{key}"], dtype=float)
        summary[key] = {
            "test_mean": float(test.mean()),
            "test_std": float(test.std(ddof=1)) if test.size > 1 else 0.0,
            "train_mean": float(train.mean()),
            "train_std": float(train.std(ddof=1)) if train.size > 1 else 0.0,
        }
    logger.info(
        "%s full CV f1_macro %.4f qwk %.4f balanced_accuracy %.4f",
        name,
        summary["f1_macro"]["test_mean"],
        summary["qwk"]["test_mean"],
        summary["balanced_accuracy"]["test_mean"],
    )
    return summary


def _curve_params(name: str) -> list[tuple[str, list[Any], str]]:
    if name == "hist_gradient_boosting":
        return [
            ("clf__learning_rate", [0.03, 0.05, 0.1, 0.15, 0.25], "learning rate"),
            ("clf__max_iter", [20, 30, 50, 80, 120], "boosting iterations"),
        ]
    if name == "logistic_regression":
        return [
            ("clf__C", [0.01, 0.1, 1.0, 10.0, 100.0], "inverse regularisation C"),
            ("preprocess__title__tfidf__max_features", [500, 1500, 3000, 5000, 8000], "title max features"),
        ]
    if name == "random_forest":
        return [
            ("clf__max_depth", [4, 8, 12, 18, None], "max depth"),
            ("clf__n_estimators", [40, 80, 120, 200, 300], "trees"),
        ]
    return [
        ("clf__max_depth", [2, 4, 8, 12, 20], "max depth"),
        ("clf__ccp_alpha", [0.0, 0.0005, 0.001, 0.005, 0.01], "ccp alpha"),
    ]


def _plot_curves(
    name: str,
    params: dict[str, Any],
    frame: Any,
    y: Any,
    config: dict[str, Any],
    tuning: dict[str, Any],
) -> list[dict[str, Any]]:
    from sklearn.model_selection import StratifiedShuffleSplit

    from scripts.train import apply_params

    n_rows = int(tuning["validation_curve_rows"])
    if len(frame) > n_rows:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=n_rows, random_state=RANDOM_SEED)
        _, keep = next(splitter.split(frame, y))
        sample = frame.iloc[keep]
        sample_y = y.iloc[keep]
    else:
        sample = frame
        sample_y = y
    pipeline = make_model_pipeline(name, config, use_description=False, seed=RANDOM_SEED)
    apply_params(pipeline, params)
    cv = StratifiedKFold(n_splits=int(tuning["validation_curve_splits"]), shuffle=True, random_state=RANDOM_SEED)
    figure_dir = REPO_ROOT / "reports" / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    described: list[dict[str, Any]] = []
    for axis, (param, grid, label) in zip(axes, _curve_params(name), strict=True):
        logger.info("Validation curve %s = %s on %s rows", param, grid, len(sample))
        train_scores, valid_scores = validation_curve(
            pipeline,
            sample,
            sample_y,
            param_name=param,
            param_range=grid,
            cv=cv,
            scoring="f1_macro",
            n_jobs=int(tuning["n_jobs"]),
            error_score=np.nan,
        )
        train_mean = np.nanmean(train_scores, axis=1)
        valid_mean = np.nanmean(valid_scores, axis=1)
        train_std = np.nanstd(train_scores, axis=1)
        valid_std = np.nanstd(valid_scores, axis=1)
        positions = np.arange(len(grid))
        axis.plot(positions, train_mean, marker="o", color="#4C78A8", label="train")
        axis.fill_between(positions, train_mean - train_std, train_mean + train_std, color="#4C78A8", alpha=0.15)
        axis.plot(positions, valid_mean, marker="o", color="#E45756", label="validation")
        axis.fill_between(positions, valid_mean - valid_std, valid_mean + valid_std, color="#E45756", alpha=0.15)
        axis.set_xticks(positions, [str(item) for item in grid], rotation=30, ha="right")
        axis.set_xlabel(label)
        axis.set_ylabel("Macro-F1")
        axis.legend(frameon=False)
        gap = train_mean - valid_mean
        described.append(
            {
                "param": param,
                "label": label,
                "grid": [None if item is None else item for item in grid],
                "train_mean": [float(item) for item in train_mean],
                "valid_mean": [float(item) for item in valid_mean],
                "widest_gap": float(np.nanmax(gap)),
                "gap_at": str(grid[int(np.nanargmax(gap))]),
            }
        )
    fig.suptitle(f"{name}: train vs validation macro-F1")
    figure_path = figure_dir / "validation_curves_formulation_b.png"
    fig.savefig(figure_path, dpi=120)
    plt.close(fig)
    return described


def _fmt(block: dict[str, float] | None, key: str = "test_mean") -> str:
    if not block:
        return "n/a"
    std_key = "train_std" if key.startswith("train") else "test_std"
    return f"{block[key]:.4f} ± {block[std_key]:.4f}"


def render_report(payload: dict[str, Any]) -> str:
    tuning = payload["tuning"]
    lines = [
        "# Tuning results — Formulation B",
        "",
        "The product label is the 4-level maturity bucket. This search does not score the sealed test set.",
        "",
        "## Budget",
        "",
        f"- Search: `HalvingRandomSearchCV`, `{tuning['n_candidates']}` candidates, factor `{tuning['factor']}`.",
        f"- Resource: training rows, from `{tuning['min_resources']}` up to `{tuning['max_resources']}` (not the full pool).",
        f"- Ranking metric inside the search: macro-F1. Folds: `{tuning['cv_splits']}`, seed `{payload['seed']}`.",
        "- After the search, each family's best parameter set is scored again with a 5-fold on **all** Formulation B training rows. Those rows are the numbers in the table below.",
        f"- Primary selection metric: full-pool macro-F1. QWK and balanced accuracy are reported beside it.",
        f"- Families searched: {', '.join(f'`{name}`' for name in tuning['models'])}.",
        "- Dummies, extra trees, and ComplementNB stay at the untuned comparison in `reports/target_design.md`. Extra trees tracked random forest there, and ComplementNB was behind on QWK.",
        "- HistGradientBoosting `max_iter` in the search is 30 or 50 and `max_bins` is 32 or 64. A 255-bin histogram on the full title matrix did not finish a single formulation in 25 minutes during the comparison.",
        f"- Validation curves use a stratified subset of `{tuning['validation_curve_rows']}` training rows and `{tuning['validation_curve_splits']}` folds, so the overfitting boundary can be drawn without another full search.",
        f"- Wall clock, search plus full-pool rescoring plus curves plus final fit: **{payload['wall_clock_seconds']:.0f} seconds**.",
        "",
        "## Search spaces",
        "",
    ]
    for name, space in payload["spaces"].items():
        lines.append(f"### `{name}`")
        lines.append("")
        lines.append("| parameter | values |")
        lines.append("|---|---|")
        for param, values in space.items():
            shown = ", ".join("None" if item is None else str(item) for item in values)
            lines.append(f"| `{param}` | {shown} |")
        lines.append("")
    lines.append("## Full training-pool 5-fold scores of each family's best parameters")
    lines.append("")
    lines.append("| model | macro-F1 | balanced accuracy | QWK | train macro-F1 | halving macro-F1 | search seconds |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in payload["models"]:
        full = row.get("full_cv") or {}
        lines.append(
            "| `{model}` | {f1} | {bal} | {qwk} | {train} | {halving:.4f} ± {halving_std:.4f} | {seconds:.0f} |".format(
                model=row["model"],
                f1=_fmt(full.get("f1_macro")),
                bal=_fmt(full.get("balanced_accuracy")),
                qwk=_fmt(full.get("qwk")),
                train=_fmt(full.get("f1_macro"), "train_mean") if full.get("f1_macro") else "n/a",
                halving=row["halving_best_f1_macro"],
                halving_std=row["halving_best_f1_macro_std"],
                seconds=row["seconds"],
            )
        )
    lines.append("")
    winner = payload["winner"]
    lines.append(f"## Selected model: `{winner}`")
    lines.append("")
    lines.append("Chosen by full-pool macro-F1. The pipeline is refit on every Formulation B training row and written to `models/best_model.joblib`.")
    lines.append("")
    lines.append("| parameter | value |")
    lines.append("|---|---|")
    for key, value in payload["winner_params"].items():
        lines.append(f"| `{key}` | `{value}` |")
    lines.append("")
    lines.append("## Halving rungs")
    lines.append("")
    for row in payload["models"]:
        lines.append(f"### `{row['model']}`")
        lines.append("")
        lines.append("| rung | candidates | rows | best macro-F1 on that rung |")
        lines.append("|---:|---:|---:|---:|")
        for rung in row["rungs"]:
            best = "n/a" if rung["best_f1_macro"] is None else f"{rung['best_f1_macro']:.4f}"
            lines.append(f"| {rung['rung']} | {rung['candidates']} | {rung['resources']} | {best} |")
        lines.append("")
    lines.append("## Validation curves")
    lines.append("")
    lines.append("Figure: `reports/figures/validation_curves_formulation_b.png`.")
    lines.append("")
    lines.append(payload["curve_reading"])
    lines.append("")
    for curve in payload["curves"]:
        lines.append(f"- `{curve['param']}`: widest train−validation gap is {curve['widest_gap']:.3f} macro-F1 at `{curve['gap_at']}`.")
    lines.append("")
    lines.append("## Re-run")
    lines.append("")
    lines.append("```")
    lines.append("python scripts/tune.py")
    lines.append("python scripts/train.py")
    lines.append("```")
    lines.append("")
    lines.append("`scripts/train.py` refits the saved parameter set and does not repeat the search.")
    lines.append("")
    return "\n".join(lines)


def _curve_reading(curves: list[dict[str, Any]]) -> str:
    if not curves:
        return "Validation curves were not produced."
    bits = []
    for curve in curves:
        train = curve["train_mean"]
        valid = curve["valid_mean"]
        if valid[-1] + 0.02 < max(valid):
            bits.append(
                f"For {curve['label']}, validation macro-F1 peaks before the right-hand side of the grid and then falls, which is the overfitting boundary."
            )
        elif train[-1] - valid[-1] > train[0] - valid[0] + 0.05:
            bits.append(
                f"For {curve['label']}, the train/validation gap widens toward the flexible end of the grid ({curve['gap_at']})."
            )
        else:
            bits.append(
                f"For {curve['label']}, validation macro-F1 does not collapse at the flexible end of this grid; the gap stays about {curve['widest_gap']:.3f}."
            )
    return " ".join(bits)


def run() -> None:
    configure_logging()
    warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
    started = time.perf_counter()
    config = load_config()
    tuning = config["tuning"]
    frame, info = formulation_b_training_frame(config)
    y = frame["y"].astype(str)
    logger.info("Formulation B training rows: %s. Sealed test rows held out: %s", info["n_train"], info["n_test_sealed"])
    partial_dir = REPO_ROOT / "reports" / "tuning_runs"
    partial_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []
    for name in tuning["models"]:
        path = partial_dir / f"{name}.json"
        if path.exists():
            logger.info("Reusing %s", path)
            summaries.append(json.loads(path.read_text(encoding="utf-8")))
            continue
        summary = _search_one(name, frame, y, config, tuning)
        summary["full_cv"] = _full_cv(name, summary["best_params"], frame, y, config, int(tuning["cv_splits"]))
        path.write_text(json.dumps(_jsonable(summary), indent=2), encoding="utf-8")
        summaries.append(summary)

    winner_row = max(summaries, key=lambda row: row["full_cv"]["f1_macro"]["test_mean"])
    winner = winner_row["model"]
    curves = _plot_curves(winner, winner_row["best_params"], frame, y, config, tuning)
    payload = {
        "seed": RANDOM_SEED,
        "tuning": tuning,
        "spaces": {name: search_spaces()[name] for name in tuning["models"]},
        "models": summaries,
        "winner": winner,
        "winner_params": winner_row["best_params"],
        "curves": curves,
        "curve_reading": _curve_reading(curves),
        "wall_clock_seconds": time.perf_counter() - started,
        "n_train": info["n_train"],
        "n_test_sealed": info["n_test_sealed"],
    }
    report_path = REPO_ROOT / "reports" / "tuning_results.md"
    report_path.write_text(render_report(_jsonable(payload)), encoding="utf-8")
    params_path = REPO_ROOT / "reports" / "best_params.json"
    params_path.write_text(
        json.dumps(
            {
                "formulation": "B",
                "model": winner,
                "params": winner_row["best_params"],
                "cv_scores": winner_row["full_cv"],
                "seed": RANDOM_SEED,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("Wrote %s. Winner: %s", report_path, winner)
    fit_and_save(config)


def main() -> None:
    run()


if __name__ == "__main__":
    main()
