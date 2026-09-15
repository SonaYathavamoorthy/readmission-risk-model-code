# Hospital Readmission Risk Predictor

A machine learning model, trained end-to-end and deployed as an
interactive app, predicting 30-day inpatient readmission risk from
discharge-time patient factors — built on top of the warehouse from
the [Synthetic EHR Patient Journey Pipeline](../synthetic-ehr-pipeline).

## Business question

That earlier project calculated a single number: 17.1% of discharges
were followed by a readmission within 30 days. This project asks the
next question a health system would actually ask: ***which specific
patients*** are at elevated risk, at the moment of discharge, so care
teams can target follow-up where it matters? A single population-wide
rate tells you there's a problem; a per-patient risk score is what
lets a discharge planner act on it.

## What's different from the other two projects

- **EHR pipeline** = batch data engineering (build the warehouse)
- **Public health dashboard** = incremental data engineering (keep a live feed current)
- **This project** = machine learning + deployment (turn the warehouse into a predictive, interactive tool)

## Approach

1. **Feature engineering** (`src/features.py`) — for every inpatient
   encounter, builds features knowable *at discharge*: age, gender,
   length of stay, prior admissions in the preceding 365 days, active
   condition count, active medication count, and total claim cost.
   Deliberately excludes anything that would leak future information.
2. **Model training & comparison** (`src/train.py`) — trains a
   Logistic Regression and a Random Forest, evaluates both on a held-out
   test set (AUC, precision, recall, F1), and selects the better one.
3. **Deployment** (`src/app.py`) — a Streamlit app where you enter a
   patient's discharge factors and get a live risk score back.

## Results

| Model | AUC | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | 0.793 | 0.350 | 0.933 | 0.509 |
| **Random Forest (selected)** | **0.803** | 0.344 | 0.733 | 0.468 |

Top predictive features: total claim cost, age, medication count.

**See [MODEL_CARD.md](MODEL_CARD.md)** for the full breakdown, including
a real limitation found through testing (not glossed over): the model's
risk score actually *decreases* slightly with more prior admissions,
which contradicts real clinical literature and is almost certainly a
small-sample artifact — documented and tested explicitly rather than
hidden.

## Getting started

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 0. Copy the warehouse from the EHR pipeline project first (not included
#    here -- it's regenerated there, not duplicated in this repo):
#    cp ../synthetic-ehr-pipeline/data/processed/ehr_warehouse.db data/

# 2. Build features from the EHR warehouse
python src/features.py

# 3. Train and compare models
python src/train.py

# 4. Run the interactive app
streamlit run src/app.py

# 5. Run tests
pytest tests/ -v
```

Note: `data/ehr_warehouse.db` is a copy of the warehouse built by the
Synthetic EHR Patient Journey Pipeline project — this project reuses
that data rather than duplicating the ETL work.

## Tech stack

Python, pandas, scikit-learn, joblib, Streamlit, pytest.

## Disclaimer

Trained on synthetic data for portfolio purposes. Not validated for
clinical use — see MODEL_CARD.md for full limitations.
