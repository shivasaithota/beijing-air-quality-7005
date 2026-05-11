"""
Reusable plotting helpers for the Beijing air-quality EDA.

All functions return a matplotlib Figure (or seaborn FacetGrid) so they
can be embedded in a Jupyter notebook or a Streamlit app via
`st.pyplot(fig)`. Plotly versions are also provided where interactivity
adds real value (time series, choropleths, etc.).
"""

from __future__ import annotations

from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", context="notebook")

POLLUTANTS = ["pm2_5", "pm10", "so2", "no2", "co", "o3"]
METEO = ["temp", "pres", "dewp", "rain", "wspm"]

# Consistent colour palette for the four selected stations
STATION_PALETTE = {
    "Dongsi": "#d62728",     # urban — red
    "Guanyuan": "#ff7f0e",   # urban — orange
    "Changping": "#2ca02c",  # suburban — green
    "Huairou": "#1f77b4",    # suburban — blue
}


# ---- Univariate ------------------------------------------------------------

def plot_distributions(
    df: pd.DataFrame,
    cols: Iterable[str] = POLLUTANTS,
    by_station: bool = False,
    log_scale: bool = True,
) -> plt.Figure:
    """
    Histograms (with KDE overlay) for the given numeric columns.

    Pollutant distributions are heavily right-skewed in air-quality data,
    so a log x-axis is the default — it makes the distribution shape readable.
    """
    cols = list(cols)
    n = len(cols)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows))
    axes = np.array(axes).ravel()

    for ax, col in zip(axes, cols):
        if by_station:
            for station, sub in df.groupby("station"):
                sns.kdeplot(
                    sub[col].dropna(), ax=ax, label=station,
                    color=STATION_PALETTE.get(station), log_scale=log_scale,
                )
            ax.legend(fontsize=8)
        else:
            data = df[col].dropna()
            if log_scale:
                data = data[data > 0]
            sns.histplot(data, kde=True, ax=ax, log_scale=log_scale, color="#4c72b0")
        ax.set_title(col.upper())
        ax.set_xlabel("")

    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle("Distribution of pollutants" + (" by station" if by_station else ""),
                 fontsize=13, y=1.02)
    fig.tight_layout()
    return fig


def plot_boxplot_by_station(df: pd.DataFrame, col: str = "pm2_5") -> plt.Figure:
    """Boxplot of one pollutant grouped by station."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    order = ["Dongsi", "Guanyuan", "Changping", "Huairou"]
    sns.boxplot(
        data=df, x="station", y=col, order=order,
        palette=STATION_PALETTE, ax=ax, showfliers=False,
    )
    ax.set_title(f"{col.upper()} by station")
    ax.set_xlabel("Station")
    ax.set_ylabel(f"{col.upper()} concentration")
    fig.tight_layout()
    return fig


# ---- Bivariate -------------------------------------------------------------

def plot_scatter(df: pd.DataFrame, x: str, y: str, hue: str = "station_type",
                 sample: int = 5000) -> plt.Figure:
    """
    Scatter of two variables. Samples for performance; full data has
    >140 k rows.
    """
    if len(df) > sample:
        df = df.sample(sample, random_state=42)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(data=df, x=x, y=y, hue=hue, alpha=0.5, s=15, ax=ax)
    ax.set_title(f"{y.upper()} vs {x.upper()}")
    fig.tight_layout()
    return fig


# ---- Multivariate ----------------------------------------------------------

def plot_correlation_heatmap(df: pd.DataFrame, cols: Iterable[str] | None = None) -> plt.Figure:
    """Pearson correlation heatmap of numeric columns."""
    if cols is None:
        cols = POLLUTANTS + METEO
    corr = df[list(cols)].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, vmin=-1, vmax=1, square=True, ax=ax,
                cbar_kws={"shrink": 0.8})
    ax.set_title("Pearson correlation — pollutants & meteorology")
    fig.tight_layout()
    return fig


def plot_pairgrid(df: pd.DataFrame, cols: Iterable[str] | None = None,
                  sample: int = 3000) -> sns.PairGrid:
    """Pairplot — sampled for tractability."""
    cols = list(cols) if cols is not None else ["pm2_5", "no2", "o3", "temp", "wspm"]
    sub = df.sample(min(len(df), sample), random_state=42)[cols + ["station_type"]]
    g = sns.pairplot(sub, vars=cols, hue="station_type", diag_kind="kde",
                     plot_kws={"alpha": 0.4, "s": 10})
    g.fig.suptitle("Pairwise relationships by station type", y=1.02)
    return g


# ---- Temporal --------------------------------------------------------------

def plot_monthly_trend(df: pd.DataFrame, col: str = "pm2_5") -> plt.Figure:
    """Monthly mean of one pollutant for each station, over the full period."""
    monthly = (
        df.groupby([pd.Grouper(key="datetime", freq="ME"), "station"])[col]
        .mean().reset_index()
    )
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for station, sub in monthly.groupby("station"):
        ax.plot(sub["datetime"], sub[col], label=station,
                color=STATION_PALETTE.get(station), linewidth=1.4)
    ax.set_title(f"Monthly mean {col.upper()} by station")
    ax.set_ylabel(f"{col.upper()}")
    ax.set_xlabel("Date")
    ax.legend(loc="upper right", ncol=4, fontsize=9)
    fig.tight_layout()
    return fig


def plot_diurnal_cycle(df: pd.DataFrame, col: str = "pm2_5") -> plt.Figure:
    """Mean concentration by hour of day, faceted by station."""
    diurnal = df.groupby(["hour", "station"])[col].mean().reset_index()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for station, sub in diurnal.groupby("station"):
        ax.plot(sub["hour"], sub[col], marker="o", markersize=3,
                label=station, color=STATION_PALETTE.get(station))
    ax.set_title(f"Diurnal cycle — mean {col.upper()} by hour of day")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel(f"{col.upper()}")
    ax.set_xticks(range(0, 24, 2))
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


def plot_seasonal_boxplot(df: pd.DataFrame, col: str = "pm2_5") -> plt.Figure:
    """Boxplot of one pollutant by season, hue by station type."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    order = ["Spring", "Summer", "Autumn", "Winter"]
    sns.boxplot(data=df, x="season", y=col, hue="station_type",
                order=order, ax=ax, showfliers=False, palette="Set2")
    ax.set_title(f"Seasonal distribution — {col.upper()}")
    fig.tight_layout()
    return fig


# ---- AQI categorical -------------------------------------------------------

def plot_aqi_category_distribution(df: pd.DataFrame) -> plt.Figure:
    """Stacked bar of AQI category proportions per station."""
    order = ["Excellent", "Good", "Lightly polluted",
             "Moderately polluted", "Heavily polluted", "Severely polluted"]
    counts = (
        df.groupby(["station", "aqi_category"]).size()
        .unstack(fill_value=0).reindex(columns=order, fill_value=0)
    )
    proportions = counts.div(counts.sum(axis=1), axis=0) * 100
    fig, ax = plt.subplots(figsize=(8, 4.5))
    proportions.plot(kind="bar", stacked=True, ax=ax, colormap="RdYlGn_r",
                     width=0.7, edgecolor="white")
    ax.set_title("AQI category distribution by station")
    ax.set_ylabel("Percentage of hours (%)")
    ax.set_xlabel("Station")
    ax.legend(title="AQI category", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.xticks(rotation=0)
    fig.tight_layout()
    return fig
