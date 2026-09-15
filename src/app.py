"""
Streamlit app: enter a patient's discharge-time factors, get a 30-day
readmission risk score back. Run with:

    streamlit run src/app.py
"""
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "readmission_model.joblib"
METRICS_PATH = ROOT / "models" / "metrics.json"

st.set_page_config(page_title="Readmission Risk Predictor", page_icon="🏥")
st.title("30-Day Readmission Risk Predictor")
st.caption(
    "Trained on synthetic EHR data (Synthea). For portfolio/demo purposes only -- "
    "not a validated clinical tool. See the model card in README.md for limitations."
)

metrics = json.loads(METRICS_PATH.read_text())
model = joblib.load(MODEL_PATH)

with st.sidebar:
    st.subheader("Model info")
    st.write(f"**Model:** {metrics['selected_model']}")
    metrics_key = "random_forest" if metrics["selected_model"] == "RandomForest" else "logreg"
    st.write(f"**Test AUC:** {metrics[metrics_key]['auc']}")
    st.write(f"**Training set positive rate:** {metrics['positive_rate']*100:.1f}%")

st.subheader("Patient discharge factors")
col1, col2 = st.columns(2)
with col1:
    age = st.slider("Age at encounter", 0, 100, 55)
    gender = st.selectbox("Gender", ["Female", "Male"])
    length_of_stay = st.slider("Length of stay (days)", 0, 30, 3)
    prior_admissions = st.slider("Prior inpatient admissions (past 365 days)", 0, 10, 0)
with col2:
    condition_count = st.slider("Active conditions this encounter", 0, 20, 2)
    medication_count = st.slider("Active medications this encounter", 0, 30, 3)
    total_cost = st.number_input("Total claim cost ($)", min_value=0, value=8000, step=500)

if st.button("Predict readmission risk"):
    features = pd.DataFrame([{
        "age_at_encounter": age, "gender": 1 if gender == "Male" else 0,
        "length_of_stay_days": length_of_stay, "prior_admissions_365d": prior_admissions,
        "condition_count": condition_count, "medication_count": medication_count,
        "total_claim_cost": total_cost,
    }])
    proba = model.predict_proba(features)[0, 1]

    st.metric("30-day readmission risk", f"{proba*100:.1f}%")
    if proba >= 0.5:
        st.warning("Elevated risk -- would flag for discharge-planning follow-up in a real workflow.")
    else:
        st.success("Lower predicted risk based on the factors entered.")

    st.caption(
        "This score reflects patterns in a small synthetic dataset (356 encounters) "
        "and should not be interpreted as a real clinical risk estimate."
    )
