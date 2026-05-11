"""
Preprocessing for the merged Beijing air-quality dataset.

Covers:
  * duplicate handling
  * missing-value treatment (forward/back fill within station, then
    column-median fallback for stubborn gaps)
  * datetime feature engineering (year, month, day, hour, weekday,
    season, part-of-day)
  * Chinese AQI computation per HJ 633-2012 (the relevant standard
    for Beijing data; differs from the US EPA AQI in breakpoints)

All functions are pure: they take a DataFrame in and return a new one.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

# ---- Constants -------------------------------------------------------------

POLLUTANT_COLS = ["pm2_5", "pm10", "so2", "no2", "co", "o3"]
METEO_COLS = ["temp", "pres", "dewp", "rain", "wspm"]
NUMERIC_COLS = POLLUTANT_COLS + METEO_COLS


# ---- China IAQI breakpoints (HJ 633-2012) ----------------------------------
# Each entry: (concentration_low, concentration_high, IAQI_low, IAQI_high)
# Concentrations are in µg/m³ except CO which is in mg/m³ (the dataset
# stores CO in µg/m³, so we divide by 1000 before lookup).

_IAQI_LEVELS = [(0, 50), (50, 100), (100, 150), (150, 200),
                (200, 300), (300, 400), (400, 500)]

_BREAKPOINTS = {
    # PM2.5: 24-hour averaging (we use the hourly value as a proxy here —
    # standard simplification for hourly air-quality dashboards).
    "pm2_5": [0, 35, 75, 115, 150, 250, 350, 500],
    "pm10":  [0, 50, 150, 250, 350, 420, 500, 600],
    "so2":   [0, 50, 150, 475, 800, 1600, 2100, 2620],   # 24-hour values
    "no2":   [0, 40, 80, 180, 280, 565, 750, 940],       # 24-hour values
    "co_mg": [0, 2, 4, 14, 24, 36, 48, 60],              # CO in mg/m³
    "o3":    [0, 100, 160, 215, 265, 800, 1000, 1200],   # 1-hour values
}

AQI_CATEGORIES = [
    (0, 50, "Excellent"),
    (51, 100, "Good"),
    (101, 150, "Lightly polluted"),
    (151, 200, "Moderately polluted"),
    (201, 300, "Heavily polluted"),
    (301, 500, "Severely polluted"),
]


def _iaqi_for(value: float, breakpoints: list[float]) -> float:
    """Linearly interpolate the IAQI for a single concentration value."""
    if pd.isna(value) or value < 0:
        return np.nan
    # Cap at the top of the scale
    if value >= breakpoints[-1]:
        return 500.0
    for i, (c_lo, c_hi) in enumerate(zip(breakpoints[:-1], breakpoints[1:])):
        if c_lo <= value < c_hi:
            i_lo, i_hi = _IAQI_LEVELS[i]
            return (i_hi - i_lo) / (c_hi - c_lo) * (value - c_lo) + i_lo
    return np.nan


def _aqi_category(aqi: float) -> str:
    if pd.isna(aqi):
        return "Unknown"
    for lo, hi, label in AQI_CATEGORIES:
        if lo <= aqi <= hi:
            return label
    return "Beyond index"


# ---- Public preprocessing pipeline ----------------------------------------


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact-duplicate rows and any duplicate (station, datetime) pairs."""
    before = len(df)
    df = df.drop_duplicates()
    df = df.drop_duplicates(subset=["station", "datetime"], keep="first")
    after = len(df)
    if before != after:
        print(f"Removed {before - after} duplicate row(s)")
    return df


def fill_missing(
    df: pd.DataFrame,
    numeric_cols: Iterable[str] = NUMERIC_COLS,
) -> pd.DataFrame:
    """
    Two-pass missing-value treatment for numeric columns:

    1. Within each station, forward-fill then back-fill (preserves local
       temporal structure — air-quality readings are autocorrelated, so
       neighbouring hours are usually a better estimate than a global mean).
    2. Anything still missing (e.g. an entire opening run of NaNs) gets the
       station-specific median.

    Categorical wind direction (wd) gets the station-specific mode.
    """
    df = df.copy()
    df = df.sort_values(["station", "datetime"])

    # Pass 1: time-aware fill within each station
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df.groupby("station")[col].transform(lambda s: s.ffill().bfill())

    # Pass 2: median fallback within station
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df.groupby("station")[col].transform(
                lambda s: s.fillna(s.median())
            )

    # Wind direction — categorical
    if "wd" in df.columns:
        df["wd"] = df.groupby("station")["wd"].transform(
            lambda s: s.fillna(s.mode().iloc[0] if not s.mode().empty else "NA")
        )

    return df.reset_index(drop=True)


def add_datetime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract calendar features from the datetime column."""
    df = df.copy()
    dt = df["datetime"]
    df["year"] = dt.dt.year
    df["month"] = dt.dt.month
    df["day"] = dt.dt.day
    df["hour"] = dt.dt.hour
    df["dayofweek"] = dt.dt.dayofweek               # 0 = Monday
    df["is_weekend"] = df["dayofweek"].isin([5, 6])

    # Meteorological seasons (Northern Hemisphere)
    df["season"] = df["month"].map(
        {12: "Winter", 1: "Winter", 2: "Winter",
         3: "Spring", 4: "Spring", 5: "Spring",
         6: "Summer", 7: "Summer", 8: "Summer",
         9: "Autumn", 10: "Autumn", 11: "Autumn"}
    )

    # Part of day
    bins = [-1, 5, 11, 17, 21, 24]
    labels = ["Night", "Morning", "Afternoon", "Evening", "Night "]
    df["part_of_day"] = pd.cut(df["hour"], bins=bins, labels=labels, ordered=False)
    # Collapse the duplicate "Night " back to "Night"
    df["part_of_day"] = df["part_of_day"].astype(str).str.strip()

    return df


def add_aqi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the China AQI per HJ 633-2012.

    AQI = max(IAQI_p) over all measured pollutants p. The pollutant with
    the highest IAQI is the 'primary pollutant'. We also tag a categorical
    AQI level for downstream visual grouping.
    """
    df = df.copy()

    # Convert CO from µg/m³ to mg/m³ for the AQI lookup
    co_mg = df["co"] / 1000.0

    iaqi = pd.DataFrame({
        "pm2_5": df["pm2_5"].apply(lambda v: _iaqi_for(v, _BREAKPOINTS["pm2_5"])),
        "pm10":  df["pm10"].apply(lambda v: _iaqi_for(v, _BREAKPOINTS["pm10"])),
        "so2":   df["so2"].apply(lambda v: _iaqi_for(v, _BREAKPOINTS["so2"])),
        "no2":   df["no2"].apply(lambda v: _iaqi_for(v, _BREAKPOINTS["no2"])),
        "co":    co_mg.apply(lambda v: _iaqi_for(v, _BREAKPOINTS["co_mg"])),
        "o3":    df["o3"].apply(lambda v: _iaqi_for(v, _BREAKPOINTS["o3"])),
    })

    df["aqi"] = iaqi.max(axis=1)
    df["aqi_primary_pollutant"] = iaqi.idxmax(axis=1)
    df["aqi_category"] = df["aqi"].apply(_aqi_category)

    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full preprocessing pipeline."""
    df = remove_duplicates(df)
    df = fill_missing(df)
    df = add_datetime_features(df)
    df = add_aqi(df)
    return df


# ---- Convenience -----------------------------------------------------------

def missing_value_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per-column count and percentage of missing values."""
    n = len(df)
    summary = pd.DataFrame({
        "missing": df.isna().sum(),
        "missing_pct": df.isna().sum() / n * 100,
        "dtype": df.dtypes.astype(str),
    })
    return summary.sort_values("missing", ascending=False)
