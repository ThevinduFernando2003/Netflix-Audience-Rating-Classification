"""Refit Formulation B on the sealed training pool using the tuned parameters.

Does not score the held-out test set. Run scripts/tune.py first.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sklearn
from sklearn.pipeline import Pipeline

from src.config import RANDOM_SEED, REPO_ROOT, load_config
from src.data.dataset import formulation_b_training_frame
from src.logging_conf import configure_logging
from src.models.persistence import save_model, utc_now
from src.models.pipelines import BUCKET_ORDER, make_model_pipeline

logger = logging.getLogger(__name__)


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


def apply_params(pipeline: Pipeline, params: dict[str, Any]) -> Pipeline:
    """Restore parameters saved as JSON. ngram_range is stored as a list and must be a tuple."""
    restored: dict[str, Any] = {}
    for key, value in params.items():
        if key.endswith("ngram_range") and isinstance(value, list):
            restored[key] = tuple(value)
        else:
            restored[key] = value
    pipeline.set_params(**restored)
    return pipeline


def fit_and_save(config: dict[str, Any] | None = None) -> Path:
    configure_logging()
    config = config if config is not None else load_config()
    params_path = REPO_ROOT / "reports" / "best_params.json"
    if not params_path.exists():
        raise SystemExit("reports/best_params.json is missing. Run scripts/tune.py before scripts/train.py.")
    saved = json.loads(params_path.read_text(encoding="utf-8"))
    frame, info = formulation_b_training_frame(config)
    name = str(saved["model"])
    pipeline = make_model_pipeline(name, config, use_description=bool(info["use_description"]), seed=RANDOM_SEED)
    apply_params(pipeline, saved["params"])
    logger.info("Fitting %s on %s Formulation B training rows", name, info["n_train"])
    pipeline.fit(frame, frame["y"].astype(str))
    feature_names: list[str] = []
    try:
        feature_names = [str(item) for item in pipeline.named_steps["preprocess"].get_feature_names_out()]
    except Exception as exc:  # pragma: no cover - names are diagnostic, not required to predict
        logger.warning("Could not read feature names: %s", exc)
    metadata = {
        "model_name": name,
        "formulation": "B",
        "label_names_in_ordinal_order": BUCKET_ORDER,
        "trained_at": utc_now(),
        "git_sha": _git_sha(),
        "data_sha256": _sha256(REPO_ROOT / config["paths"]["raw_csv"]),
        "n_train": info["n_train"],
        "n_test_sealed_unscored": info["n_test_sealed"],
        "class_counts_train": info["class_counts"],
        "random_seed": RANDOM_SEED,
        "library_versions": {"python": sys.version.split()[0], "scikit-learn": sklearn.__version__},
        "params": saved["params"],
        "cv_scores": saved.get("cv_scores", {}),
        "feature_names": feature_names,
        "test_set_scored": False,
    }
    model_path = REPO_ROOT / "models" / "best_model.joblib"
    metadata_path = REPO_ROOT / "models" / "metadata.json"
    save_model(pipeline, model_path, metadata, metadata_path)
    logger.info("Wrote %s", model_path)
    return model_path


def main() -> None:
    fit_and_save()


if __name__ == "__main__":
    main()
