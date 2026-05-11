"""Page 1 \u2014 Dataset explorer."""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import data_loader, preprocessing  # noqa: E402

st.set_page_config(page_title="Dataset \u2014 Beijing AQ", page_icon="\U0001F4C2", layout="wide")
st.title("\U0001F4C2 Dataset")
st.caption("Inspect the raw and preprocessed Beijing air-quality dataset.")


@st.cache_data(show_spinner="Loading merged dataset\u2026")
def get_raw() -> pd.DataFrame:
    return data_loader.load_merged_dataset()


@st.cache_data(show_spinner="Running preprocessing pipeline\u2026")
def get_processed() -> pd.DataFrame:
    return preprocessing.preprocess(get_raw())


# ---- Tabs ------------------------------------------------------------------

tab_overview, tab_raw, tab_processed, tab_quality = st.tabs(
    ["Overview", "Raw sample", "Preprocessed sample", "Data quality"]
)

with tab_overview:
    df = get_raw()
    st.subheader("Schema and shape")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Columns", df.shape[1])
    c3.metric("Stations", df["station"].nunique())

    st.subheader("Date coverage per station")
    coverage = (
        df.groupby("station")["datetime"]
        .agg(start="min", end="max", n_hours="count").reset_index()
    )
    st.dataframe(coverage, use_container_width=True)

    st.subheader("Column descriptions")
    st.markdown(
        """
| Column | Description | Units |
| --- | --- | --- |
| `datetime` | Hourly timestamp | \u2014 |
| `station` | Monitoring station name | \u2014 |
| `station_type` | Urban / suburban tag | \u2014 |
| `pm2_5`, `pm10` | Particulate matter | \u00b5g/m\u00b3 |
| `so2`, `no2`, `o3` | Gas pollutants | \u00b5g/m\u00b3 |
| `co` | Carbon monoxide | \u00b5g/m\u00b3 |
| `temp` | Air temperature | \u00b0C |
| `pres` | Atmospheric pressure | hPa |
| `dewp` | Dew-point temperature | \u00b0C |
| `rain` | Hourly rainfall | mm |
| `wd` | Wind direction (16-point) | \u2014 |
| `wspm` | Wind speed | m/s |
        """
    )

with tab_raw:
    df = get_raw()
    st.subheader("Raw data preview")
    station = st.selectbox("Station", sorted(df["station"].unique()), key="raw_station")
    n = st.slider("Rows to show", 5, 500, 50, 5, key="raw_n")
    st.dataframe(df[df["station"] == station].head(n), use_container_width=True)

    st.subheader("Statistical summary (numeric columns)")
    st.dataframe(df.describe().T.round(2), use_container_width=True)

with tab_processed:
    df = get_processed()
    st.subheader("Preprocessed data preview")
    station = st.selectbox("Station", sorted(df["station"].unique()), key="pp_station")
    n = st.slider("Rows to show", 5, 500, 50, 5, key="pp_n")
    st.dataframe(df[df["station"] == station].head(n), use_container_width=True)

    st.subheader("Engineered features")
    st.markdown(
        "- **Datetime features**: `year`, `month`, `day`, `hour`, "
        "`dayofweek`, `is_weekend`, `season`, `part_of_day`\n"
        "- **AQI features**: `aqi`, `aqi_primary_pollutant`, `aqi_category` "
        "(China HJ 633-2012 standard)"
    )

    st.subheader("AQI category counts")
    cat_counts = df["aqi_category"].value_counts().reset_index()
    cat_counts.columns = ["AQI category", "Count"]
    st.dataframe(cat_counts, use_container_width=True)

with tab_quality:
    raw = get_raw()
    proc = get_processed()

    st.subheader("Missing values \u2014 before vs after preprocessing")
    before = preprocessing.missing_value_summary(raw).rename(
        columns={"missing": "missing_before", "missing_pct": "pct_before"}
    )[["missing_before", "pct_before"]]
    after = preprocessing.missing_value_summary(proc).rename(
        columns={"missing": "missing_after", "missing_pct": "pct_after"}
    )[["missing_after", "pct_after"]]
    comparison = before.join(after, how="outer").fillna(0).round(2)
    st.dataframe(comparison, use_container_width=True)

    st.subheader("Duplicate check")
    n_exact = raw.duplicated().sum()
    n_keys = raw.duplicated(subset=["station", "datetime"]).sum()
    c1, c2 = st.columns(2)
    c1.metric("Exact duplicate rows", f"{n_exact:,}")
    c2.metric("Duplicate (station, datetime) pairs", f"{n_keys:,}")
