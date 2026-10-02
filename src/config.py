"""Project configuration. RANDOM_SEED is the single seed for every split and model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

RANDOM_SEED: int = 42

REPO_ROOT: Path = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH: Path = REPO_ROOT / "config.yaml"


def load_config(path: Path | None = None) -> dict[str, Any]:
    """Load config.yaml and refuse to run if its seed disagrees with RANDOM_SEED."""
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with config_path.open(encoding="utf-8") as handle:
        config: dict[str, Any] = yaml.safe_load(handle)
    seed = int(config["seed"])
    if seed != RANDOM_SEED:
        raise ValueError(
            f"config.yaml seed is {seed}, but src.config.RANDOM_SEED is {RANDOM_SEED}. "
            "Keep them identical so two runs cannot silently diverge."
        )
    config["seed"] = seed
    return config


def resolve_path(config: dict[str, Any], key: str) -> Path:
    """Resolve a config path relative to the repository root."""
    raw = config["paths"][key]
    path = Path(raw)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path
