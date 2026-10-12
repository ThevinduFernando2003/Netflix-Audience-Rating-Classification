"""Capture local screenshots and a demo video of the Streamlit page and the API docs.

Requires the app on port 8502, the API on port 8001, and Playwright with Microsoft Edge.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "demo"
UI = "http://127.0.0.1:8502"
API_DOCS = "http://127.0.0.1:8001/docs"


def _fill(page, label: str, value: str) -> None:
    field = page.get_by_label(label, exact=True)
    field.click()
    field.fill(value)
    page.wait_for_timeout(250)


def _choose_type(page, name: str) -> None:
    page.get_by_label("Type", exact=True).click()
    page.get_by_role("option", name=name).click()
    page.wait_for_timeout(250)


def _show_result(page) -> None:
    """Streamlit scrolls an inner pane, so move the prediction to the top of that pane."""
    page.evaluate(
        """() => {
          const heading = [...document.querySelectorAll('h3')].at(-1);
          const main = document.querySelector('[data-testid="stMain"]');
          if (heading && main) {
            const top = heading.getBoundingClientRect().top - main.getBoundingClientRect().top + main.scrollTop;
            main.scrollTop = Math.max(0, top - 16);
          } else {
            heading?.scrollIntoView({block: 'start'});
          }
        }"""
    )
    page.wait_for_timeout(700)


def _predict(page, heading: str) -> None:
    page.get_by_role("button", name="Predict").click()
    page.get_by_role("heading", name=heading, exact=True).wait_for(timeout=180_000)
    page.locator('[data-testid="stVegaLiteChart"], [data-testid="stDataFrame"]').first.wait_for(timeout=30_000)
    _show_result(page)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    raw_video = OUT / "_video"
    if raw_video.exists():
        shutil.rmtree(raw_video)
    raw_video.mkdir()

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="msedge", headless=True, slow_mo=60)
        context = browser.new_context(
            viewport={"width": 1280, "height": 1400},
            record_video_dir=str(raw_video),
            record_video_size={"width": 1280, "height": 1400},
        )
        page = context.new_page()
        page.goto(UI, wait_until="domcontentloaded")
        page.get_by_role("heading", name="Audience rating").wait_for(timeout=30_000)
        page.wait_for_timeout(800)
        page.screenshot(path=str(OUT / "01_empty_form.png"), full_page=True)

        _fill(page, "Title", "Peppa Pig")
        _choose_type(page, "TV Show")
        _fill(page, "Genres", "Kids' TV")
        _fill(page, "Duration", "2 Seasons")
        _fill(page, "Country", "United Kingdom")
        _fill(page, "Release year", "2004")
        page.wait_for_timeout(500)
        page.screenshot(path=str(OUT / "02_kids_form.png"), full_page=True)
        _predict(page, "Kids")
        page.screenshot(path=str(OUT / "03_kids_warning.png"))

        _fill(page, "Title", "The Godfather")
        _choose_type(page, "Movie")
        _fill(page, "Genres", "Classic Movies, Dramas")
        _fill(page, "Duration", "175 min")
        _fill(page, "Director", "Francis Ford Coppola")
        _fill(page, "Country", "United States")
        _fill(page, "Release year", "1972")
        page.wait_for_timeout(500)
        _predict(page, "Mature")
        page.screenshot(path=str(OUT / "04_mature_result.png"))

        page.goto(API_DOCS, wait_until="domcontentloaded")
        page.get_by_text("Netflix audience rating").first.wait_for(timeout=30_000)
        page.wait_for_timeout(1000)
        page.screenshot(path=str(OUT / "05_api_docs.png"), full_page=True)

        video = page.video
        context.close()
        browser.close()
        if video is not None:
            saved = Path(video.path())
            target = OUT / "demo.webm"
            if target.exists():
                target.unlink()
            shutil.move(str(saved), target)
    shutil.rmtree(raw_video, ignore_errors=True)
    print("Wrote", OUT)


if __name__ == "__main__":
    main()
