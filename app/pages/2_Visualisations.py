"""Page 2 \u2014 Visualisations."""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import data_loader, eda, preprocessing  # noqa: E402

st.set_page_config(page_title="Visualisations \u2014 Beijing AQ",
                   page_icon="\U0001F4CA", layout="wide")
st.title("\U0001F4CA Visualisations")
st.caption("Interactive exploration of pollutants, meteorology and temporal patterns.")


@st.cache_data(show_spinner="Loading processed dataset\u2026")
def get_data() -> pd.DataFrame:
    return preprocessing.preprocess(data_loader.load_merged_dataset())


df = get_data()

# ---- Sidebar filters -------------------------------------------------------

with st.sidebar:
    st.header("Filters")
    stations = st.multiselect(
        "Stations", sorted(df["station"].unique()),
        default=sorted(df["station"].unique()),
    )
    date_range = st.date_input(
        "Date range",
        value=(df["datetime"].min().date(), df["datetime"].max().date()),
        min_value=df["datetime"].min().date(),
        max_value=df["datetime"].max().date(),
    )

if not stations:
    st.warning("Select at least one station from the sidebar.")
    st.stop()

mask = df["station"].isin(stations)
if isinstance(date_range, tuple) and len(date_range) == 2:
    start, end = date_range
    mask &= (df["datetime"].dt.date >= start) & (df["datetime"].dt.date <= end)
fdf = df[mask]

st.caption(f"Filtered subset: {len(fdf):,} rows across {fdf['station'].nunique()} station(s).")


# ---- Tabs ------------------------------------------------------------------

tab_uni, tab_bi, tab_multi, tab_temporal, tab_aqi = st.tabs(
    ["Univariate", "Bivariate", "Multivariate", "Temporal", "AQI"]
)

with tab_uni:
    c1, c2 = st.columns([1, 3])
    with c1:
        col = st.selectbox("Variable", eda.POLLUTANTS + eda.METEO,
                           index=0, key="uni_col")
        by_station = st.checkbox("Compare by station (KDE overlay)", value=True)
        log_scale = st.checkbox("Log-scale x-axis", value=True)
    with c2:
        if by_station:
            fig = eda.plot_distributions(fdf, cols=[col], by_station=True, log_scale=log_scale)
        else:
            fig = eda.plot_distributions(fdf, cols=[col], by_station=False, log_scale=log_scale)
        st.pyplot(fig)

    st.subheader("By-station boxplot")
    box_col = st.selectbox("Variable for boxplot",
                           eda.POLLUTANTS + eda.METEO, index=0, key="box_col")
    st.pyplot(eda.plot_boxplot_by_station(fdf, box_col))

with tab_bi:
    st.subheader("Scatter plot")
    c1, c2, c3 = st.columns(3)
    with c1:
        x = st.selectbox("X axis", eda.POLLUTANTS + eda.METEO, index=6, key="bi_x")
    with c2:
        y = st.selectbox("Y axis", eda.POLLUTANTS + eda.METEO, index=0, key="bi_y")
    with c3:
        hue = st.selectbox("Colour by", ["station_type", "station", "season"],
                           index=0, key="bi_hue")
    st.pyplot(eda.plot_scatter(fdf, x, y, hue=hue))

with tab_multi:
    st.subheader("Correlation heatmap")
    cols = st.multiselect(
        "Variables to include",
        eda.POLLUTANTS + eda.METEO,
        default=eda.POLLUTANTS + eda.METEO,
    )
    if len(cols) >= 2:
        st.pyplot(eda.plot_correlation_heatmap(fdf, cols))
    else:
        st.info("Select at least two variables.")

with tab_temporal:
    var = st.selectbox("Variable", eda.POLLUTANTS, index=0, key="temp_var")

    st.subheader(f"Monthly mean {var.upper()}")
    st.pyplot(eda.plot_monthly_trend(fdf, var))

    st.subheader(f"Diurnal cycle \u2014 mean {var.upper()} by hour of day")
    st.pyplot(eda.plot_diurnal_cycle(fdf, var))

    st.subheader(f"Seasonal distribution \u2014 {var.upper()}")
    st.pyplot(eda.plot_seasonal_boxplot(fdf, var))

with tab_aqi:
    st.subheader("AQI category distribution by station")
    st.pyplot(eda.plot_aqi_category_distribution(fdf))

    st.subheader("Primary pollutant frequency")
    primary = (
        fdf["aqi_primary_pollutant"].value_counts(normalize=True) * 100
    ).round(1).reset_index()
    primary.columns = ["Primary pollutant", "% of hours"]
    st.dataframe(primary, use_container_width=True)
