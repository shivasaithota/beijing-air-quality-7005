"""
Beijing Air Quality \u2014 Streamlit application.

Run from the project root:
    streamlit run app/streamlit_app.py

The app uses Streamlit's native multi-page support: any .py file dropped
into `app/pages/` becomes a navigable page in the sidebar.
"""

import sys
from pathlib import Path

import streamlit as st

# Allow `from src import ...` regardless of where Streamlit was launched
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


st.set_page_config(
    page_title="Beijing Air Quality \u2014 PRAC1",
    page_icon="\U0001F30D",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main() -> None:
    st.title("\U0001F30D Beijing Air Quality \u2014 Data to Application")
    st.caption("CMP7005 PRAC1 \u2014 Programming for Data Analysis")

    st.markdown(
        """
This interactive application accompanies the CMP7005 PRAC1 submission.
It explores hourly air-quality measurements from four monitoring stations
in Beijing (1 March 2013 \u2013 28 February 2017), compares pollution
patterns across urban and suburban sites, and lets you generate
short-horizon PM2.5 forecasts from a trained Random Forest model.

### \U0001F4C2 Pages

Use the sidebar to navigate:

| Page | What you can do |
| --- | --- |
| **1. Dataset** | Inspect the raw and preprocessed data, summary statistics, missing-value patterns, and the AQI feature engineering. |
| **2. Visualisations** | Build interactive charts: distributions, correlations, monthly trends, diurnal cycles, AQI breakdowns. |
| **3. Model** | Review model performance, feature importances, and run live PM2.5 predictions on user-supplied conditions. |

### \U0001F4DA Stations included

- **Urban:** Dongsi, Guanyuan
- **Suburban:** Changping, Huairou

### \U0001F527 First run

If you haven't yet built the merged dataset or trained the model, do so
once from the project root:

```bash
python -m src.data_loader     # downloads + merges the four station CSVs
jupyter lab notebooks/CMP7005_PRAC1_analysis.ipynb   # runs EDA + trains the model
```

The app loads the cached dataset and model artefacts on demand.
        """
    )

    st.divider()

    col1, col2, col3 = st.columns(3)
    col1.metric("Stations", "4", "2 urban + 2 suburban")
    col2.metric("Time period", "2013\u20132017", "4 years, hourly")
    col3.metric("Total records", "~140k", "after merging")


if __name__ == "__main__" or True:
    main()
