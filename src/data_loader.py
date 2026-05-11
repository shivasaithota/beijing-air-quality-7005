"""
Data loading for the Beijing Multi-Site Air Quality dataset.

Downloads the UCI archive (12 station CSVs, hourly, 1 March 2013 –
28 February 2017), extracts the four selected stations (2 urban + 2
suburban), parses timestamps, and merges them into a single tidy
DataFrame for downstream analysis.

Run as a script:
    python -m src.data_loader

This will populate `data/raw/` with the 4 station CSVs and write the
merged dataset to `data/processed/merged.parquet`.
"""

from __future__ import annotations

import io
import logging
import zipfile
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

# ---- Configuration ---------------------------------------------------------

# UCI archive URL for the Beijing Multi-Site Air Quality dataset.
UCI_ZIP_URL = (
    "https://archive.ics.uci.edu/static/public/501/"
    "beijing+multi+site+air+quality+data.zip"
)

# Selected stations — see report Section 1 for justification.
# Two central urban stations + two distant suburban/rural stations.
URBAN_STATIONS = ["Dongsi", "Guanyuan"]
SUBURBAN_STATIONS = ["Changping", "Huairou"]
SELECTED_STATIONS = URBAN_STATIONS + SUBURBAN_STATIONS

# Project paths — anchored relative to this file so the script works
# from any working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MERGED_PATH = PROCESSED_DIR / "merged.parquet"


# ---- Public API ------------------------------------------------------------


def download_dataset(target_dir: Path = RAW_DIR, force: bool = False) -> list[Path]:
    """
    Download the UCI archive and extract the four selected station CSVs.

    Parameters
    ----------
    target_dir : Path
        Directory where station CSVs will be written.
    force : bool
        Re-download even if target files already exist.

    Returns
    -------
    list[Path]
        Paths to the extracted station CSV files.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    expected_files = [
        target_dir / f"PRSA_Data_{station}_20130301-20170228.csv"
        for station in SELECTED_STATIONS
    ]

    if not force and all(p.exists() for p in expected_files):
        logger.info("All selected station CSVs already exist in %s", target_dir)
        return expected_files

    logger.info("Downloading UCI archive (~40 MB) from %s", UCI_ZIP_URL)
    response = requests.get(UCI_ZIP_URL, timeout=120)
    response.raise_for_status()

    # The UCI download is a zip-of-zip: outer zip contains
    # `PRSA2017_Data_20130301-20170228.zip`, which contains the per-station CSVs.
    with zipfile.ZipFile(io.BytesIO(response.content)) as outer_zip:
        # Find the inner zip
        inner_zip_name = next(
            (n for n in outer_zip.namelist() if n.lower().endswith(".zip")),
            None,
        )
        if inner_zip_name is None:
            # Some mirrors flatten the structure — try CSVs directly
            inner_archive = outer_zip
        else:
            with outer_zip.open(inner_zip_name) as inner_bytes:
                inner_archive = zipfile.ZipFile(io.BytesIO(inner_bytes.read()))

        with inner_archive:
            extracted: list[Path] = []
            for station in SELECTED_STATIONS:
                pattern = f"PRSA_Data_{station}_"
                csv_name = next(
                    (n for n in inner_archive.namelist() if pattern in n and n.endswith(".csv")),
                    None,
                )
                if csv_name is None:
                    raise FileNotFoundError(
                        f"Could not find station '{station}' in the UCI archive."
                    )
                out_path = target_dir / Path(csv_name).name
                with inner_archive.open(csv_name) as src, open(out_path, "wb") as dst:
                    dst.write(src.read())
                logger.info("  extracted %s", out_path.name)
                extracted.append(out_path)

    return extracted


def load_station(csv_path: Path) -> pd.DataFrame:
    """
    Load a single station CSV and add a proper datetime index.

    The raw schema is:
        No, year, month, day, hour, PM2.5, PM10, SO2, NO2, CO, O3,
        TEMP, PRES, DEWP, RAIN, wd, WSPM, station
    """
    df = pd.read_csv(csv_path)

    # Combine the year/month/day/hour columns into a single datetime
    df["datetime"] = pd.to_datetime(
        df[["year", "month", "day", "hour"]].rename(columns={"hour": "hour"}),
        errors="raise",
    )

    # Drop the now-redundant columns and the row index column
    df = df.drop(columns=["No", "year", "month", "day", "hour"])

    # Standardise column names (lowercase, snake-case for meteorology)
    df = df.rename(
        columns={
            "PM2.5": "pm2_5",
            "PM10": "pm10",
            "SO2": "so2",
            "NO2": "no2",
            "CO": "co",
            "O3": "o3",
            "TEMP": "temp",
            "PRES": "pres",
            "DEWP": "dewp",
            "RAIN": "rain",
            "WSPM": "wspm",
            "wd": "wd",
            "station": "station",
        }
    )

    # Reorder for readability
    cols = ["datetime", "station",
            "pm2_5", "pm10", "so2", "no2", "co", "o3",
            "temp", "pres", "dewp", "rain", "wd", "wspm"]
    df = df[cols]

    return df


def merge_stations(csv_paths: Iterable[Path]) -> pd.DataFrame:
    """Concatenate per-station dataframes and tag urban/suburban category."""
    frames = [load_station(p) for p in csv_paths]
    merged = pd.concat(frames, ignore_index=True)

    # Tag each row with its station type (urban vs suburban) — useful for EDA
    merged["station_type"] = merged["station"].map(
        {**{s: "urban" for s in URBAN_STATIONS},
         **{s: "suburban" for s in SUBURBAN_STATIONS}}
    )

    # Sort by station then time for predictable ordering
    merged = merged.sort_values(["station", "datetime"]).reset_index(drop=True)

    logger.info(
        "Merged %d rows across %d stations (%s)",
        len(merged), merged["station"].nunique(),
        ", ".join(sorted(merged["station"].unique())),
    )
    return merged


def build_merged_dataset(force_download: bool = False) -> pd.DataFrame:
    """End-to-end: download (if needed), load, merge, and persist to parquet."""
    csv_paths = download_dataset(force=force_download)
    merged = merge_stations(csv_paths)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(MERGED_PATH, index=False)
    logger.info("Wrote merged dataset to %s (%.1f MB)",
                MERGED_PATH, MERGED_PATH.stat().st_size / 1e6)

    return merged


def load_merged_dataset() -> pd.DataFrame:
    """Load the cached merged dataset, building it on first run."""
    if MERGED_PATH.exists():
        return pd.read_parquet(MERGED_PATH)
    return build_merged_dataset()


# ---- CLI entry point -------------------------------------------------------

if __name__ == "__main__":
    df = build_merged_dataset()
    print("\nFirst rows:")
    print(df.head())
    print("\nShape:", df.shape)
    print("\nStations:", df["station"].value_counts().to_dict())
