# CMP7005 PRAC1 — From Data to Application Development

**Module:** CMP7005 Programming for Data Analysis
**Assessment:** PRAC1 (70%)
**Topic:** Beijing PM2.5 air-quality analysis and forecasting

End-to-end Python project covering data ingestion, exploratory data analysis,
predictive modelling, and an interactive Streamlit application for the Beijing
Multi-Site Air Quality dataset (12 nationally controlled monitoring stations,
1 March 2013 – 28 February 2017).

## Selected stations

Two urban (within Beijing's central ring roads) and two suburban stations are
analysed, following the categorisation used by Yao et al. (2015) and
Xu & Zhang (2020):

| Station | Type | Notes |
| --- | --- | --- |
| Dongsi | Urban | Central, inside 2nd Ring Road |
| Guanyuan | Urban | Central-west, inside 2nd Ring Road |
| Changping | Suburban | Northern district, ~30 km from centre |
| Huairou | Suburban | Far north-east, mountainous, ~50 km from centre |

Justification is detailed in the report (Section 1).

## Project layout

```
beijing-air-quality/
├── README.md                        # This file
├── requirements.txt                 # Python dependencies
├── .gitignore
├── data/
│   ├── raw/                         # Downloaded per-station CSVs
│   └── processed/                   # Cleaned, merged dataset
├── notebooks/
│   └── CMP7005_PRAC1_analysis.ipynb # Main analysis notebook (EDA + model)
├── src/                             # Reusable modules
│   ├── __init__.py
│   ├── data_loader.py               # Download + merge stations
│   ├── preprocessing.py             # Cleaning, AQI, feature engineering
│   ├── eda.py                       # Plot helpers
│   └── model.py                     # Train/evaluate predictive model
├── app/
│   ├── streamlit_app.py             # App entry point
│   └── pages/
│       ├── 1_Dataset.py             # Dataset explorer
│       ├── 2_Visualisations.py      # Interactive charts
│       └── 3_Model.py               # Model predictions + evaluation
├── models/                          # Saved model artefacts
├── report/                          # Final report (Word + PDF)
└── screenshots/                     # GitHub commit/repo screenshots for report
```

## Setup

Requires Python 3.10+.

```bash
# Clone the repo
git clone https://github.com/<your-username>/beijing-air-quality.git
cd beijing-air-quality

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate          # macOS/Linux
# .venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

## Running the project

### 1. Download and merge the data

```bash
python -m src.data_loader
```

This fetches the four selected station CSVs from the UCI Machine Learning
Repository, saves them to `data/raw/`, and writes the merged dataset to
`data/processed/merged.parquet`.

### 2. Run the analysis notebook

```bash
jupyter lab notebooks/CMP7005_PRAC1_analysis.ipynb
```

The notebook walks through Tasks 1–3: data understanding, preprocessing,
EDA, and model training. It saves the trained model to `models/`.

### 3. Launch the Streamlit app

```bash
streamlit run app/streamlit_app.py
```

The app opens in the browser with three pages: **Dataset**, **Visualisations**,
and **Model**.

## Deliverables checklist

- [x] Task 1 — Data handling (`src/data_loader.py`, notebook §1)
- [x] Task 2 — EDA (notebook §2–4, `src/preprocessing.py`, `src/eda.py`)
- [x] Task 3 — Model building (notebook §5, `src/model.py`)
- [x] Task 4 — Streamlit application (`app/`)
- [x] Task 5 — Version control (this repo + screenshots in `screenshots/`)
- [x] Report (`report/`)

## AI acknowledgement

This assignment uses AI tools as permitted under the "AI Acknowledged" category
of the assessment brief. A statement detailing the support used and how it was
integrated is provided at the end of the report.

## References

- Yao, L. et al. (2015). *Sources Apportionment of PM2.5 in a Background Site
  in the North China Plain.* IJERPH, 12(10), 12264–12286.
- Xu, J. & Zhang, X. (2020). *Foliar uptake of atmospheric mercury by Quercus
  variabilis in urban Beijing*. J. Environmental Management, 264.
- UCI Machine Learning Repository — Beijing Multi-Site Air-Quality Data Set:
  https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data
