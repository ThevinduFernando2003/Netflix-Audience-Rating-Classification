"""Page for one Formulation B prediction. Run from the repository root."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.serving.predict import load_saved_model, model_path, score_title

st.set_page_config(page_title="Audience rating", layout="centered")
st.title("Audience rating")
st.caption("Formulation B maturity buckets, in order: Kids, Older Kids, Teens, Mature.")

with st.form("title"):
    title = st.text_input("Title")
    kind = st.selectbox("Type", ["Movie", "TV Show"])
    listed_in = st.text_input("Genres", placeholder="Dramas, Independent Movies")
    duration = st.text_input("Duration", placeholder="90 min or 2 Seasons")
    director = st.text_input("Director")
    country = st.text_input("Country")
    date_added = st.text_input("Date added", placeholder="January 1, 2020")
    release_year = st.text_input("Release year")
    submitted = st.form_submit_button("Predict")

if not submitted:
    st.stop()

if not title.strip():
    st.error("Enter a title.")
    st.stop()

if not model_path().exists():
    st.error(f"Saved model not found at {model_path()}.")
    st.stop()

year: int | None
text_year = release_year.strip()
if not text_year:
    year = None
elif text_year.isdigit():
    year = int(text_year)
else:
    st.error("Release year must be a number.")
    st.stop()

result = score_title(
    load_saved_model(),
    load_config(),
    {
        "title": title,
        "type": kind,
        "listed_in": listed_in,
        "duration": duration,
        "director": director,
        "country": country,
        "date_added": date_added,
        "release_year": year,
    },
)

st.subheader(result["prediction"])
if result["severe_error_warning"]:
    st.error(result["severe_error_warning"])

chart = pd.DataFrame(result["buckets"])
st.bar_chart(chart.set_index("name")["probability"], horizontal=True)
st.dataframe(
    chart.assign(probability=chart["probability"].map(lambda value: f"{value:.1%}")),
    hide_index=True,
    use_container_width=True,
)
