"""Hand-written maturity lexicons. Counts are exact token matches, not substrings."""

from __future__ import annotations

import re
from pathlib import Path

_TOKEN = re.compile(r"[a-z0-9]+")


def load_lexicon(path: Path) -> set[str]:
    """Read one word per line. Blank lines and # comments are ignored."""
    terms: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip().lower()
        if not stripped or stripped.startswith("#"):
            continue
        terms.add(stripped)
    if not terms:
        raise ValueError(f"Lexicon {path} has no terms.")
    return terms


def load_lexicons(paths: dict[str, str], root: Path) -> dict[str, set[str]]:
    loaded: dict[str, set[str]] = {}
    for name, relative in paths.items():
        path = Path(relative)
        if not path.is_absolute():
            path = root / path
        loaded[name] = load_lexicon(path)
    return loaded


def tokenize(text: str) -> set[str]:
    return set(_TOKEN.findall(text.lower()))


def lexicon_counts(text: str, lexicons: dict[str, set[str]]) -> dict[str, int]:
    """Count how many lexicon terms occur as whole tokens. Phrases match as substrings of the lowered text."""
    lowered = text.lower()
    tokens = tokenize(lowered)
    counts: dict[str, int] = {}
    for name, terms in lexicons.items():
        hits = 0
        for term in terms:
            if " " in term:
                hits += int(term in lowered)
            else:
                hits += int(term in tokens)
        counts[name] = hits
    return counts
