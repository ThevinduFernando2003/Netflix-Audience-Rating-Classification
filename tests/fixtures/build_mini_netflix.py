"""Build the committed fixture. The CSV is the artifact tests load; this file records how each required case was chosen."""

from __future__ import annotations

import csv
from pathlib import Path

COLUMNS = [
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


def _row(index: int, **overrides: str) -> dict[str, str]:
    base = {
        "show_id": f"m{index:03d}",
        "type": "Movie",
        "title": f"Story {index}",
        "director": "Ada Lovelace",
        "cast": "Actor One, Actor Two",
        "country": "United States",
        "date_added": "September 25, 2021",
        "release_year": "2019",
        "rating": "PG",
        "duration": "90 min",
        "listed_in": "Dramas",
        "description": "A family adventure at school.",
    }
    base.update(overrides)
    return base


def build_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = [
        _row(1, rating="74 min", duration="", title="Duration In Rating", description="A stand-up special."),
        _row(
            2,
            rating="74 min",
            duration="90 min",
            title="Duration Already Set",
            description="A stand-up special with a runtime already recorded.",
        ),
        _row(3, rating="", title="Missing Rating", description="The catalogue never certified this title."),
        _row(4, rating="NR", title="Not Rated Story", description="An unrated drama."),
        _row(5, rating="UR", title="Unrated Story", description="An unrated comedy."),
        _row(6, rating="NC-17", title="Explicit Story", description="A mature explicit drama about murder."),
        _row(7, director="", title="No Director", description="A family adventure."),
        _row(8, cast="", title="No Cast", description="A family adventure."),
        _row(9, country="", title="No Country", description="A family adventure."),
        _row(
            10,
            country="United States, India",
            title="Two Countries",
            description="A family adventure across two countries.",
        ),
        _row(
            11,
            listed_in="Kids' TV, Dramas, Comedies, Thrillers, Documentaries",
            title="Five Genres",
            type="TV Show",
            duration="1 Season",
            rating="TV-Y",
            description="A family adventure for children.",
        ),
        _row(12, title="Amélie", director="Raúl Campos", description="A family adventure in Paris."),
        _row(13, title="Hello, World", description="A family adventure with a comma in the title."),
        _row(14, director="Not Given", country="Not Given", title="Sentinel Missing", description="A family adventure."),
        _row(
            15,
            type="TV Show",
            duration="2 Seasons",
            rating="TV-MA",
            listed_in="TV Dramas, Crime TV Shows",
            title="Long Crime Show",
            description="A murder and drug crime thriller for adults.",
        ),
        _row(16, date_added="not-a-date", title="Bad Date", description="A family adventure."),
        _row(
            17,
            title="Lexicon Check",
            description="A family faces murder and drug crime in school",
            rating="TV-MA",
            listed_in="Thrillers",
        ),
        _row(
            18,
            director="Raúl Campos, Jan Suter",
            cast="A, B, C",
            country="United States, India",
            date_added="September 25, 2021",
            release_year="2018",
            type="TV Show",
            duration="2 Seasons",
            rating="TV-Y",
            listed_in="Kids' TV, Dramas, Comedies, Thrillers, Documentaries",
            title="Amélie, A Tale",
            description="A family faces murder and drug crime in school",
        ),
    ]
    specs = [
        ("TV-Y", "Kids", "Children & Family Movies", "A children family adventure at school."),
        ("TV-G", "Kids", "Kids' TV", "A children family cartoon adventure."),
        ("G", "Kids", "Children & Family Movies", "A children family adventure."),
        ("TV-Y7", "Older Kids", "Kids' TV", "A children adventure for older kids."),
        ("TV-Y7-FV", "Older Kids", "Kids' TV", "A children adventure with fantasy."),
        ("PG", "Older Kids", "Children & Family Movies, Comedies", "A family adventure comedy."),
        ("TV-PG", "Older Kids", "Kids' TV, TV Comedies", "A family adventure comedy."),
        ("PG-13", "Teens", "Dramas, Comedies", "A teen drama about school friends."),
        ("TV-14", "Teens", "Teen TV Shows, TV Dramas", "A teen drama."),
        ("R", "Mature", "Thrillers, Horror Movies", "A murder thriller for adults."),
        ("TV-MA", "Mature", "TV Dramas, Crime TV Shows", "A murder and drug crime story for adults."),
        ("NC-17", "Mature", "Dramas, Independent Movies", "An explicit adult drama."),
    ]
    next_id = 19
    # Seven of each certificate: more than four titles in every maturity bucket, and enough rows to stratify.
    for _ in range(7):
        for rating, _bucket, genres, description in specs:
            is_tv = rating.startswith("TV-") or "TV" in genres
            rows.append(
                _row(
                    next_id,
                    rating=rating,
                    type="TV Show" if is_tv else "Movie",
                    duration="1 Season" if is_tv else "95 min",
                    listed_in=genres,
                    description=description,
                    title=f"Story {next_id} {rating}",
                    date_added="9/25/2021" if next_id % 2 == 0 else "2020-01-15",
                )
            )
            next_id += 1
    return rows


def main() -> None:
    rows = build_rows()
    path = Path(__file__).with_name("mini_netflix.csv")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {path}")


if __name__ == "__main__":
    main()
