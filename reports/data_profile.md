# Data profile

Source file: `data/raw/netflix_titles.csv`
SHA-256: `995a3774af8551d09abd5d2b2a2844a4c0cda28af083176282e42112fb1d200f`
Rows: **8790**
Columns (10): `show_id`, `type`, `title`, `director`, `country`, `date_added`, `release_year`, `rating`, `duration`, `listed_in`

This report is written by `scripts/profile_data.py`. Re-run that script to refresh it.

## How this file differs from the brief

The modelling brief describes an 8,791-row, 12-column Netflix catalogue with `cast` and `description`. The file in this repository is the source of truth and it does not match that description.

- Row count is 8790, not 8,791.
- Missing columns: `cast`, `description`.
- `description` is the brief's main text signal. It is not in this file, so no synopsis vectorizer is fitted.
- `director` and `country` use the sentinel `Not Given` instead of nulls. That sentinel is treated as missing. It is not renamed to Unknown.
- `country` is a single value in this file (no comma-separated lists). The cleaner still splits on commas so a later file with multiple countries keeps working.
- `date_added` mixes `M/D/YYYY` text with ISO dates. The parser tries an explicit format list and coerces failures to missing.

## Dtypes

| column | dtype | blank or null | blank or null % |
|---|---|---:|---:|
| `show_id` | object | 0 | 0.00% |
| `type` | object | 0 | 0.00% |
| `title` | object | 0 | 0.00% |
| `director` | object | 0 | 0.00% |
| `country` | object | 0 | 0.00% |
| `date_added` | object | 0 | 0.00% |
| `release_year` | object | 0 | 0.00% |
| `rating` | object | 0 | 0.00% |
| `duration` | object | 0 | 0.00% |
| `listed_in` | object | 0 | 0.00% |

## Duplicate show_id

Duplicate `show_id` rows: **0**.
Distinct `show_id`: **8790**.

## Missing sentinels

These are not nulls. They are explicit placeholder strings.

| column | token | count | % |
|---|---|---:|---:|
| `director` | `Not Given` | 2588 | 29.44% |
| `country` | `Not Given` | 287 | 3.27% |
| `cast` | (column absent) | 0 | 0.00% |
| `description` | (column absent) | 0 | 0.00% |

## rating value counts

Sorted by count, descending. This is the raw column, before any folding.

| rating | count | % | certification |
|---|---:|---:|---|
| `TV-MA` | 3205 | 36.46% | tv |
| `TV-14` | 2157 | 24.54% | tv |
| `TV-PG` | 861 | 9.80% | tv |
| `R` | 799 | 9.09% | film |
| `PG-13` | 490 | 5.57% | film |
| `TV-Y7` | 333 | 3.79% | tv |
| `TV-Y` | 306 | 3.48% | tv |
| `PG` | 287 | 3.27% | film |
| `TV-G` | 220 | 2.50% | tv |
| `NR` | 79 | 0.90% | unrated |
| `G` | 41 | 0.47% | film |
| `TV-Y7-FV` | 6 | 0.07% | tv |
| `NC-17` | 3 | 0.03% | film |
| `UR` | 3 | 0.03% | unrated |

Rows whose `rating` matches `^\d+\s*min$`: **0**.
Rows with a blank `rating`: **0**.
Both groups are excluded from training by `src/data/clean.py`. On this file both counts are the raw counts above.

## rating × type

| rating | Movie | TV Show | All |
|---|---:|---:|---:|
| `TV-MA` | 2062 | 1143 | 3205 |
| `TV-14` | 1427 | 730 | 2157 |
| `TV-PG` | 540 | 321 | 861 |
| `R` | 797 | 2 | 799 |
| `PG-13` | 490 | 0 | 490 |
| `TV-Y7` | 139 | 194 | 333 |
| `TV-Y` | 131 | 175 | 306 |
| `PG` | 287 | 0 | 287 |
| `TV-G` | 126 | 94 | 220 |
| `NR` | 75 | 4 | 79 |
| `G` | 41 | 0 | 41 |
| `TV-Y7-FV` | 5 | 1 | 6 |
| `NC-17` | 3 | 0 | 3 |
| `UR` | 3 | 0 | 3 |
| `All` | 6126 | 2664 | 8790 |

Film certificates and TV certificates are not separated by `type`. Most movies in this file carry a TV rating. Formulation C in `reports/target_design.md` has to deal with that.

## listed_in vocabulary

Distinct genres: **42**.
Rows with more than one genre: **6778**.

| genre | titles |
|---|---:|
| International Movies | 2752 |
| Dramas | 2426 |
| Comedies | 1674 |
| International TV Shows | 1349 |
| Documentaries | 869 |
| Action & Adventure | 859 |
| TV Dramas | 762 |
| Independent Movies | 756 |
| Children & Family Movies | 641 |
| Romantic Movies | 616 |
| Thrillers | 577 |
| TV Comedies | 573 |
| Crime TV Shows | 469 |
| Kids' TV | 448 |
| Docuseries | 394 |
| Music & Musicals | 375 |
| Romantic TV Shows | 370 |
| Horror Movies | 357 |
| Stand-Up Comedy | 343 |
| Reality TV | 255 |
| British TV Shows | 252 |
| Sci-Fi & Fantasy | 243 |
| Sports Movies | 219 |
| Anime Series | 174 |
| Spanish-Language TV Shows | 173 |
| TV Action & Adventure | 167 |
| Korean TV Shows | 151 |
| Classic Movies | 116 |
| LGBTQ Movies | 102 |
| TV Mysteries | 98 |
| Science & Nature TV | 92 |
| TV Sci-Fi & Fantasy | 83 |
| TV Horror | 75 |
| Anime Features | 71 |
| Cult Movies | 71 |
| Teen TV Shows | 69 |
| Faith & Spirituality | 65 |
| TV Thrillers | 57 |
| Stand-Up Comedy & Talk Shows | 56 |
| Movies | 53 |
| Classic & Cult TV | 26 |
| TV Shows | 16 |

## release_year

Parse failures: **0**.
Min 1925, median 2017, mean 2014.18, max 2021.

| decade | titles |
|---|---:|
| 1920s | 1 |
| 1940s | 15 |
| 1950s | 11 |
| 1960s | 25 |
| 1970s | 70 |
| 1980s | 129 |
| 1990s | 274 |
| 2000s | 807 |
| 2010s | 5913 |
| 2020s | 1545 |

## date_added parse failures

Non-blank values: **8790**.
Parse failures: **0** (0.00% of rows).

Formats accepted, in order: `%m/%d/%Y`, `%Y-%m-%d`, `%Y-%m-%d %H:%M:%S`, `%B %d, %Y`.

## duration parse failures

Parsed as minutes: **6126**.
Parsed as seasons: **2664**.
Blank duration: **0**.
Non-blank parse failures: **0**.

Accepted patterns: `^(\d+)\s*min$` and `^(\d+)\s*seasons?$`, case-insensitive.

