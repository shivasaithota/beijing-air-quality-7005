"""Page 3 \u2014 Model outputs and live predictions."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import data_loader, model, preprocessing  # noqa: E402

st.set_page_config(page_title="Model \u2014 Beijing AQ",
                   page_icon="\U0001F916", layout="wide")
st.title("\U0001F916 Model")
st.caption("Random Forest nowcast for next-hour PM2.5.")


@st.cache_data(show_spinner="Preparing data\u2026")
def _prepared() -> pd.DataFrame:
    return preprocessing.preprocess(data_loader.load_merged_dataset())


@st.cache_resource(show_spinner="Loading or training model\u2026")
def _load_or_train():
    """Try to load the saved artefact; fall back to training on the fly."""
    if model.MODEL_PATH.exists():
        return {"trained_now": False, "model": model.load_model(),
                "results": None}
    df = _prepared()
    results = model.train_and_evaluate(df, tune=False)
    return {"trained_now": True, "model": results["rf_model"], "results": results}


@st.cache_data(show_spinner="Evaluating on hold-out set\u2026")
def _evaluate_holdout() -> dict:
    """Compute baseline + RF metrics on the time-series hold-out."""
    df = _prepared()
    sup = model.build_supervised_frame(df)
    train, test = model.time_series_split(sup)
    feature_cols = model.NUMERIC_FEATURES + model.CATEGORICAL_FEATURES
    X_tr, y_tr = train[feature_cols], train[model.TARGET]
    X_te, y_te = test[feature_cols], test[model.TARGET]

    baseline = model.build_baseline_pipeline().fit(X_tr, y_tr)
    rf_pipe = _load_or_train()["model"]
    return {
        "baseline_metrics": model.evaluate(baseline, X_te, y_te),
        "rf_metrics": model.evaluate(rf_pipe, X_te, y_te),
        "feature_importances": model.feature_importances(rf_pipe),
        "test_predictions": pd.DataFrame({
            "datetime": test["datetime"].values,
            "station": test["station"].values,
            "actual": y_te.values,
            "predicted": rf_pipe.predict(X_te),
        }),
    }


df = _prepared()
artefacts = _load_or_train()
trained_pipe = artefacts["model"]
metrics = _evaluate_holdout()

# ---- Tabs ------------------------------------------------------------------

tab_perf, tab_imp, tab_predict = st.tabs(
    ["Performance", "Feature importances", "Predict"]
)

with tab_perf:
    st.subheader("Hold-out performance")
    st.caption("Metrics computed on the last 20 % of each station's series "
               "(time-aware split, no leakage).")

    base = metrics["baseline_metrics"]
    rf = metrics["rf_metrics"]

    c1, c2, c3 = st.columns(3)
    c1.metric("RMSE (\u00b5g/m\u00b3)", f"{rf['RMSE']:.2f}",
              delta=f"{rf['RMSE'] - base['RMSE']:+.2f} vs baseline",
              delta_color="inverse")
    c2.metric("MAE (\u00b5g/m\u00b3)", f"{rf['MAE']:.2f}",
              delta=f"{rf['MAE'] - base['MAE']:+.2f} vs baseline",
              delta_color="inverse")
    c3.metric("R\u00b2", f"{rf['R2']:.3f}",
              delta=f"{rf['R2'] - base['R2']:+.3f} vs baseline")

    st.markdown("**Comparison table**")
    st.dataframe(pd.DataFrame({
        "Linear baseline": base, "Random Forest": rf,
    }).round(3), use_container_width=True)

    st.subheader("Predicted vs actual (test set)")
    preds = metrics["test_predictions"]
    sample = preds.sample(min(len(preds), 4000), random_state=42)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.scatter(sample["actual"], sample["predicted"], alpha=0.3, s=10)
    lim = float(max(sample["actual"].max(), sample["predicted"].max()))
    ax.plot([0, lim], [0, lim], "r--", linewidth=1)
    ax.set_xlabel("Actual PM2.5 (\u00b5g/m\u00b3)")
    ax.set_ylabel("Predicted PM2.5 (\u00b5g/m\u00b3)")
    ax.set_title("Predicted vs actual")
    st.pyplot(fig)

    st.subheader("Two-week slice")
    station_demo = st.selectbox("Station for time-series view",
                                sorted(preds["station"].unique()))
    sub = (preds[preds["station"] == station_demo]
           .sort_values("datetime").iloc[:24 * 14])
    fig2, ax2 = plt.subplots(figsize=(11, 4))
    ax2.plot(sub["datetime"], sub["actual"], label="Actual", linewidth=1.2)
    ax2.plot(sub["datetime"], sub["predicted"], label="Predicted",
             linewidth=1.2, alpha=0.85)
    ax2.set_ylabel("PM2.5 (\u00b5g/m\u00b3)")
    ax2.legend()
    ax2.set_title(f"{station_demo} \u2014 first 2 weeks of test set")
    st.pyplot(fig2)

with tab_imp:
    st.subheader("Top features")
    imp = metrics["feature_importances"]
    st.dataframe(imp, use_container_width=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    fi = imp.iloc[::-1]
    ax.barh(fi["feature"], fi["importance"], color="#4c72b0")
    ax.set_xlabel("Importance")
    ax.set_title("Random Forest feature importances")
    st.pyplot(fig)

with tab_predict:
    st.subheader("Run a live prediction")
    st.caption("Adjust the inputs and the model will forecast next-hour PM2.5.")

    # Pre-fill defaults from a recent row in the dataset
    recent = df.sort_values("datetime").iloc[-100:]
    defaults = recent.median(numeric_only=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Station / time**")
        station = st.selectbox("Station", sorted(df["station"].unique()))
        hour = st.slider("Hour of day", 0, 23, int(defaults["hour"]))
        month = st.slider("Month", 1, 12, int(defaults["month"]))
        dayofweek = st.slider("Day of week (0 = Mon)", 0, 6, int(defaults["dayofweek"]))
        season = st.selectbox("Season", ["Spring", "Summer", "Autumn", "Winter"])
        wd = st.selectbox("Wind direction",
                          sorted(df["wd"].dropna().unique()))

    with c2:
        st.markdown("**Pollutants (current hour)**")
        pm10 = st.number_input("PM10 (\u00b5g/m\u00b3)", 0.0, 1000.0, float(defaults["pm10"]))
        so2 = st.number_input("SO\u2082 (\u00b5g/m\u00b3)", 0.0, 500.0, float(defaults["so2"]))
        no2 = st.number_input("NO\u2082 (\u00b5g/m\u00b3)", 0.0, 500.0, float(defaults["no2"]))
        co = st.number_input("CO (\u00b5g/m\u00b3)", 0.0, 10000.0, float(defaults["co"]))
        o3 = st.number_input("O\u2083 (\u00b5g/m\u00b3)", 0.0, 500.0, float(defaults["o3"]))

    with c3:
        st.markdown("**Meteorology + lagged PM2.5**")
        temp = st.number_input("Temperature (\u00b0C)", -30.0, 45.0, float(defaults["temp"]))
        pres = st.number_input("Pressure (hPa)", 950.0, 1050.0, float(defaults["pres"]))
        dewp = st.number_input("Dew point (\u00b0C)", -40.0, 35.0, float(defaults["dewp"]))
        rain = st.number_input("Rain (mm/h)", 0.0, 100.0, float(defaults["rain"]))
        wspm = st.number_input("Wind speed (m/s)", 0.0, 20.0, float(defaults["wspm"]))
        pm25_lag1 = st.number_input("PM2.5 1h ago", 0.0, 1000.0,
                                    float(defaults["pm2_5"]))
        pm25_lag3 = st.number_input("PM2.5 3h ago", 0.0, 1000.0,
                                    float(defaults["pm2_5"]))
        pm25_lag24 = st.number_input("PM2.5 24h ago", 0.0, 1000.0,
                                     float(defaults["pm2_5"]))

    if st.button("Predict next-hour PM2.5", type="primary"):
        row = pd.DataFrame([{
            "pm2_5_lag1": pm25_lag1, "pm2_5_lag3": pm25_lag3,
            "pm2_5_lag24": pm25_lag24,
            "temp": temp, "pres": pres, "dewp": dewp, "rain": rain, "wspm": wspm,
            "pm10": pm10, "so2": so2, "no2": no2, "co": co, "o3": o3,
            "hour": hour, "month": month, "dayofweek": dayofweek,
            "station": station, "wd": wd, "season": season,
        }])
        prediction = float(trained_pipe.predict(row)[0])
        st.success(f"**Predicted PM2.5 (next hour): {prediction:.1f} \u00b5g/m\u00b3**")

        # Categorise the prediction
        from src.preprocessing import _aqi_category, _iaqi_for, _BREAKPOINTS
        iaqi_pm25 = _iaqi_for(prediction, _BREAKPOINTS["pm2_5"])
        st.info(f"That corresponds to a PM2.5 IAQI of **{iaqi_pm25:.0f}** "
                f"(\u201c{_aqi_category(iaqi_pm25)}\u201d under HJ 633-2012).")
