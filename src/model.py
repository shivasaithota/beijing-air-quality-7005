"""
Model building for PM2.5 forecasting.

Target:    next-hour PM2.5 concentration (regression task)
Features:  current-hour pollutants + meteorology + lagged PM2.5
           (1, 3, 24 hour lags) + datetime features + station

Approach:
  * Time-aware train/test split (last 20% of each station's series held out)
  * sklearn `Pipeline` + `ColumnTransformer` so scaling and one-hot encoding
    happen inside cross-validation (no leakage)
  * Two models compared honestly:
        - LinearRegression baseline
        - RandomForestRegressor (final model, lightly tuned)
  * Metrics: RMSE, MAE, R²

Run from the notebook (or `python -m src.model`).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)

# ---- Configuration ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "pm25_rf.joblib"

TARGET = "pm2_5_next"
LAGGED_FEATURES = ["pm2_5_lag1", "pm2_5_lag3", "pm2_5_lag24"]
METEO_FEATURES = ["temp", "pres", "dewp", "rain", "wspm"]
POLLUTANT_FEATURES = ["pm10", "so2", "no2", "co", "o3"]
DATETIME_FEATURES = ["hour", "month", "dayofweek"]
CATEGORICAL_FEATURES = ["station", "wd", "season"]

NUMERIC_FEATURES = (
    LAGGED_FEATURES + METEO_FEATURES + POLLUTANT_FEATURES + DATETIME_FEATURES
)


# ---- Feature engineering ---------------------------------------------------

def build_supervised_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the preprocessed long-form dataframe into a supervised
    learning frame: each row predicts the *next hour's* PM2.5 at the same
    station, given the current hour's measurements + lagged history.

    Lags are computed within station to avoid bleeding values across
    locations. The first 24 rows of each station are dropped (no lag-24
    available) and the last row of each station is dropped (no target).
    """
    df = df.sort_values(["station", "datetime"]).copy()

    # Lagged PM2.5 at 1h, 3h, 24h (within station)
    g = df.groupby("station", group_keys=False)
    df["pm2_5_lag1"] = g["pm2_5"].shift(1)
    df["pm2_5_lag3"] = g["pm2_5"].shift(3)
    df["pm2_5_lag24"] = g["pm2_5"].shift(24)

    # Target: next hour's PM2.5
    df[TARGET] = g["pm2_5"].shift(-1)

    # Drop rows with NaN target or lag features (the per-station endpoints)
    df = df.dropna(subset=LAGGED_FEATURES + [TARGET]).reset_index(drop=True)
    return df


def time_series_split(
    df: pd.DataFrame, test_fraction: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Hold out the last `test_fraction` of each station's time series.
    This mirrors deployment: we forecast the future, not random rows.
    """
    train_parts, test_parts = [], []
    for _, sub in df.groupby("station"):
        sub = sub.sort_values("datetime")
        cut = int(len(sub) * (1 - test_fraction))
        train_parts.append(sub.iloc[:cut])
        test_parts.append(sub.iloc[cut:])
    return (
        pd.concat(train_parts).reset_index(drop=True),
        pd.concat(test_parts).reset_index(drop=True),
    )


# ---- Pipeline construction -------------------------------------------------

def build_preprocessor() -> ColumnTransformer:
    """Standard-scale numerics; one-hot encode categoricals."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             CATEGORICAL_FEATURES),
        ]
    )


def build_baseline_pipeline() -> Pipeline:
    """LinearRegression baseline — sets the bar to beat."""
    return Pipeline([
        ("preprocess", build_preprocessor()),
        ("model", LinearRegression()),
    ])


def build_rf_pipeline(**rf_kwargs: Any) -> Pipeline:
    """RandomForest pipeline with sensible defaults."""
    defaults = dict(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42,
    )
    defaults.update(rf_kwargs)
    return Pipeline([
        ("preprocess", build_preprocessor()),
        ("model", RandomForestRegressor(**defaults)),
    ])


# ---- Training & evaluation -------------------------------------------------

def evaluate(model: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    """RMSE, MAE, R²."""
    preds = model.predict(X)
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y, preds))),
        "MAE": float(mean_absolute_error(y, preds)),
        "R2": float(r2_score(y, preds)),
    }


def tune_random_forest(
    X_train: pd.DataFrame, y_train: pd.Series, n_splits: int = 3
) -> Pipeline:
    """
    Light grid search over a few key RF hyperparameters using a
    TimeSeriesSplit (respects temporal ordering).

    The grid is intentionally small so this finishes in a few minutes
    on a laptop. Expand it if you have more compute.
    """
    pipe = build_rf_pipeline()
    param_grid = {
        "model__n_estimators": [150, 300],
        "model__max_depth": [None, 20],
        "model__min_samples_leaf": [1, 2],
    }
    cv = TimeSeriesSplit(n_splits=n_splits)
    search = GridSearchCV(
        pipe, param_grid, cv=cv,
        scoring="neg_root_mean_squared_error", n_jobs=-1, verbose=1,
    )
    search.fit(X_train, y_train)
    logger.info("Best params: %s", search.best_params_)
    logger.info("Best CV RMSE: %.3f", -search.best_score_)
    return search.best_estimator_


def feature_importances(model: Pipeline, top_n: int = 15) -> pd.DataFrame:
    """Extract feature importances from a fitted RF pipeline."""
    rf = model.named_steps["model"]
    pre = model.named_steps["preprocess"]
    feature_names = pre.get_feature_names_out()
    importances = pd.DataFrame({
        "feature": feature_names,
        "importance": rf.feature_importances_,
    }).sort_values("importance", ascending=False).head(top_n)
    return importances.reset_index(drop=True)


# ---- Persistence -----------------------------------------------------------

def save_model(model: Pipeline, path: Path = MODEL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    logger.info("Saved model to %s", path)


def load_model(path: Path = MODEL_PATH) -> Pipeline:
    return joblib.load(path)


# ---- Convenience driver ----------------------------------------------------

def train_and_evaluate(df: pd.DataFrame, tune: bool = False) -> dict[str, Any]:
    """End-to-end: feature engineer → split → fit baseline + RF → evaluate."""
    sup = build_supervised_frame(df)
    train, test = time_series_split(sup)

    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X_tr, y_tr = train[feature_cols], train[TARGET]
    X_te, y_te = test[feature_cols], test[TARGET]

    # Baseline
    baseline = build_baseline_pipeline().fit(X_tr, y_tr)
    base_metrics = evaluate(baseline, X_te, y_te)
    logger.info("Baseline (LinearRegression): %s", base_metrics)

    # Random Forest
    if tune:
        rf = tune_random_forest(X_tr, y_tr)
    else:
        rf = build_rf_pipeline().fit(X_tr, y_tr)
    rf_metrics = evaluate(rf, X_te, y_te)
    logger.info("RandomForest: %s", rf_metrics)

    save_model(rf)

    return {
        "baseline_metrics": base_metrics,
        "rf_metrics": rf_metrics,
        "rf_model": rf,
        "feature_importances": feature_importances(rf),
        "test_predictions": pd.DataFrame({
            "datetime": test["datetime"].values,
            "station": test["station"].values,
            "actual": y_te.values,
            "predicted": rf.predict(X_te),
        }),
    }


if __name__ == "__main__":
    from src.data_loader import load_merged_dataset
    from src.preprocessing import preprocess

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    df = preprocess(load_merged_dataset())
    results = train_and_evaluate(df, tune=False)
    print("\nBaseline:", results["baseline_metrics"])
    print("RF:      ", results["rf_metrics"])
    print("\nTop features:")
    print(results["feature_importances"])
