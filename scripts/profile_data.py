"""Profile the raw catalogue and write reports/data_profile.md. Every number comes from the file."""

from __future__ import annotations

import hashlib
import logging
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import REPO_ROOT, load_config, resolve_path
from src.data.clean import is_contaminated_rating, is_missing_text, parse_date_added, parse_duration
from src.data.labels import certification_system
from src.logging_conf import configure_logging

logger = logging.getLogger(__name__)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _blank_count(series: pd.Series) -> int:
    return int(series.map(lambda value: is_missing_text(value) or str(value).strip() == "").sum())


def write_profile(frame: pd.DataFrame, source: Path, config: dict) -> str:
    """Build the markdown report from `frame` as read from disk."""
    n_rows = len(frame)
    lines: list[str] = []
    lines.append("# Data profile")
    lines.append("")
    try:
        shown = source.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        shown = source.as_posix()
    lines.append(f"Source file: `{shown}`")
    lines.append(f"SHA-256: `{_sha256(source)}`")
    lines.append(f"Rows: **{n_rows}**")
    lines.append(f"Columns ({len(frame.columns)}): {', '.join(f'`{column}`' for column in frame.columns)}")
    lines.append("")
    lines.append("This report is written by `scripts/profile_data.py`. Re-run that script to refresh it.")
    lines.append("")
    lines.append("## How this file differs from the brief")
    lines.append("")
    expected = [
        "show_id",
        "type",
        "title",
        "director",
        "cast",
        "country",
        "date_added",
        "release_year",
        "rating",
        "duration",
        "listed_in",
        "description",
    ]
    missing = [column for column in expected if column not in frame.columns]
    extra = [column for column in frame.columns if column not in expected]
    lines.append(
        "The modelling brief describes an 8,791-row, 12-column Netflix catalogue with `cast` and `description`. "
        "The file in this repository is the source of truth and it does not match that description."
    )
    lines.append("")
    lines.append(f"- Row count is {n_rows}, not 8,791.")
    if missing:
        lines.append(f"- Missing columns: {', '.join(f'`{column}`' for column in missing)}.")
    else:
        lines.append("- No expected column is missing.")
    if extra:
        lines.append(f"- Unexpected columns: {', '.join(f'`{column}`' for column in extra)}.")
    lines.append(
        "- `description` is the brief's main text signal. It is not in this file, so no synopsis vectorizer is fitted."
    )
    lines.append(
        "- `director` and `country` use the sentinel `Not Given` instead of nulls. "
        "That sentinel is treated as missing. It is not renamed to Unknown."
    )
    lines.append(
        "- `country` is a single value in this file (no comma-separated lists). "
        "The cleaner still splits on commas so a later file with multiple countries keeps working."
    )
    lines.append(
        "- `date_added` mixes `M/D/YYYY` text with ISO dates. "
        "The parser tries an explicit format list and coerces failures to missing."
    )
    lines.append("")

    lines.append("## Dtypes")
    lines.append("")
    lines.append("| column | dtype | blank or null | blank or null % |")
    lines.append("|---|---|---:|---:|")
    for column in frame.columns:
        blank = _blank_count(frame[column])
        lines.append(f"| `{column}` | {frame[column].dtype} | {blank} | {100 * blank / n_rows:.2f}% |")
    lines.append("")

    lines.append("## Duplicate show_id")
    lines.append("")
    duplicate_rows = int(frame["show_id"].duplicated().sum())
    lines.append(f"Duplicate `show_id` rows: **{duplicate_rows}**.")
    lines.append(f"Distinct `show_id`: **{frame['show_id'].nunique()}**.")
    lines.append("")

    lines.append("## Missing sentinels")
    lines.append("")
    lines.append("These are not nulls. They are explicit placeholder strings.")
    lines.append("")
    lines.append("| column | token | count | % |")
    lines.append("|---|---|---:|---:|")
    for column in ["director", "country", "cast", "description"]:
        if column not in frame.columns:
            lines.append(f"| `{column}` | (column absent) | 0 | 0.00% |")
            continue
        counts = frame[column].astype(str).str.strip().value_counts()
        for token in ["Not Given", "Unknown", ""]:
            count = int(counts.get(token, 0))
            if count:
                lines.append(f"| `{column}` | `{token or '(blank)'}` | {count} | {100 * count / n_rows:.2f}% |")
    lines.append("")

    lines.append("## rating value counts")
    lines.append("")
    lines.append("Sorted by count, descending. This is the raw column, before any folding.")
    lines.append("")
    lines.append("| rating | count | % | certification |")
    lines.append("|---|---:|---:|---|")
    rating_counts = frame["rating"].astype(str).str.strip().value_counts(dropna=False)
    for rating, count in rating_counts.items():
        token = str(rating)
        system = certification_system(token, config) if token not in {"", "nan"} else "missing"
        if is_contaminated_rating(token, config):
            system = "contaminated_duration"
        lines.append(f"| `{token}` | {int(count)} | {100 * int(count) / n_rows:.2f}% | {system} |")
    lines.append("")

    contaminated = int(frame["rating"].map(lambda value: is_contaminated_rating(value, config)).sum())
    missing_rating = int(frame["rating"].map(is_missing_text).sum())
    lines.append(f"Rows whose `rating` matches `{config['rating_contamination_pattern']}`: **{contaminated}**.")
    lines.append(f"Rows with a blank `rating`: **{missing_rating}**.")
    lines.append("Both groups are excluded from training by `src/data/clean.py`. On this file both counts are the raw counts above.")
    lines.append("")

    lines.append("## rating × type")
    lines.append("")
    crosstab = pd.crosstab(frame["rating"].astype(str).str.strip(), frame["type"].astype(str).str.strip(), margins=True)
    crosstab = crosstab.reindex(index=list(rating_counts.index) + ["All"])
    header = "| rating | " + " | ".join(str(column) for column in crosstab.columns) + " |"
    sep = "|---|" + "|".join("---:" for _ in crosstab.columns) + "|"
    lines.append(header)
    lines.append(sep)
    for rating, row in crosstab.iterrows():
        cells = " | ".join(str(int(value)) for value in row.tolist())
        lines.append(f"| `{rating}` | {cells} |")
    lines.append("")
    lines.append(
        "Film certificates and TV certificates are not separated by `type`. "
        "Most movies in this file carry a TV rating. Formulation C in `reports/target_design.md` has to deal with that."
    )
    lines.append("")

    lines.append("## listed_in vocabulary")
    lines.append("")
    genre_counts: Counter[str] = Counter()
    for cell in frame["listed_in"].astype(str):
        for part in cell.split(","):
            genre = part.strip()
            if genre:
                genre_counts[genre] += 1
    lines.append(f"Distinct genres: **{len(genre_counts)}**.")
    lines.append(f"Rows with more than one genre: **{int(frame['listed_in'].astype(str).str.contains(',').sum())}**.")
    lines.append("")
    lines.append("| genre | titles |")
    lines.append("|---|---:|")
    for genre, count in genre_counts.most_common():
        lines.append(f"| {genre} | {count} |")
    lines.append("")

    lines.append("## release_year")
    lines.append("")
    years = pd.to_numeric(frame["release_year"], errors="coerce")
    lines.append(f"Parse failures: **{int(years.isna().sum())}**.")
    lines.append(
        f"Min {int(years.min())}, median {years.median():.0f}, mean {years.mean():.2f}, max {int(years.max())}."
    )
    lines.append("")
    lines.append("| decade | titles |")
    lines.append("|---|---:|")
    decades = (years.dropna() // 10 * 10).astype(int).value_counts().sort_index()
    for decade, count in decades.items():
        lines.append(f"| {int(decade)}s | {int(count)} |")
    lines.append("")

    lines.append("## date_added parse failures")
    lines.append("")
    parsed = frame["date_added"].map(lambda value: parse_date_added(value, config["date_formats"]))
    present = ~frame["date_added"].map(is_missing_text)
    failures = frame.loc[present & parsed.isna(), "date_added"].astype(str)
    lines.append(f"Non-blank values: **{int(present.sum())}**.")
    lines.append(f"Parse failures: **{len(failures)}** ({100 * len(failures) / n_rows:.2f}% of rows).")
    if len(failures):
        lines.append("")
        lines.append("Failure values (up to 20):")
        lines.append("")
        for value, count in failures.value_counts().head(20).items():
            lines.append(f"- `{value}` × {int(count)}")
    lines.append("")
    lines.append("Formats accepted, in order: " + ", ".join(f"`{fmt}`" for fmt in config["date_formats"]) + ".")
    lines.append("")

    lines.append("## duration parse failures")
    lines.append("")
    parsed_duration = frame["duration"].map(parse_duration)
    minutes_ok = sum(not pd.isna(item[0]) for item in parsed_duration)
    seasons_ok = sum(not pd.isna(item[1]) for item in parsed_duration)
    present_duration = ~frame["duration"].map(is_missing_text)
    duration_fail = sum(
        present and pd.isna(minutes) and pd.isna(seasons)
        for present, (minutes, seasons) in zip(present_duration, parsed_duration, strict=True)
    )
    lines.append(f"Parsed as minutes: **{minutes_ok}**.")
    lines.append(f"Parsed as seasons: **{seasons_ok}**.")
    lines.append(f"Blank duration: **{int((~present_duration).sum())}**.")
    lines.append(f"Non-blank parse failures: **{duration_fail}**.")
    lines.append("")
    lines.append("Accepted patterns: `^(\\d+)\\s*min$` and `^(\\d+)\\s*seasons?$`, case-insensitive.")
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    configure_logging()
    config = load_config()
    source = resolve_path(config, "raw_csv")
    if not source.exists():
        raise SystemExit(f"Raw catalogue not found at {source}. Export it before profiling.")
    frame = pd.read_csv(source, dtype=str, keep_default_na=False, encoding="utf-8")
    report = write_profile(frame, source, config)
    output = REPO_ROOT / "reports" / "data_profile.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    logger.info("Wrote %s", output)


if __name__ == "__main__":
    main()
