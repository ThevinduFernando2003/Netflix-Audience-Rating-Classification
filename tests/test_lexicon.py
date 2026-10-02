"""Lexicon counts against a hand-checked sentence."""

from pathlib import Path

from src.features.lexicon import lexicon_counts, load_lexicon


def test_hand_checked_sentence():
    root = Path(__file__).resolve().parents[1] / "data" / "lexicons"
    lexicons = {
        "violence": load_lexicon(root / "violence.txt"),
        "sexual": load_lexicon(root / "sexual.txt"),
        "substance": load_lexicon(root / "substance.txt"),
        "family": load_lexicon(root / "family.txt"),
    }
    counts = lexicon_counts("A family faces murder and drug crime in school", lexicons)
    assert counts == {"violence": 2, "sexual": 0, "substance": 1, "family": 2}


def test_substring_is_not_a_hit():
    counts = lexicon_counts("the murderer waited", {"violence": {"murder"}})
    assert counts["violence"] == 0
