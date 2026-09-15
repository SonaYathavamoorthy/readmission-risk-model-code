"""
Feature engineering: builds a patient-level, encounter-level feature
table from the EHR warehouse (built in the Synthetic EHR Patient
Journey Pipeline project) to predict 30-day inpatient readmission.

For every inpatient encounter, we engineer features known BEFORE
discharge (so the model reflects what's actually knowable at the
point of a real discharge-planning decision) and label it 1 if the
patient had another inpatient encounter starting within 30 days.

Usage: python src/features.py
"""
import logging
import sqlite3
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "ehr_warehouse.db"
FEATURES_PATH = ROOT / "data" / "features.csv"


def build_features() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)

    inpatient = pd.read_sql("""
        SELECT encounter_id, patient_id, start_date_key, stop_date_key,
               total_claim_cost
        FROM fact_encounters
        WHERE encounter_class = 'inpatient'
        ORDER BY patient_id, start_date_key
    """, conn, parse_dates=["start_date_key", "stop_date_key"])

    patients = pd.read_sql("""
        SELECT patient_id, birthdate, gender, is_deceased,
               healthcare_expenses, healthcare_coverage
        FROM dim_patient
    """, conn, parse_dates=["birthdate"])

    # Label: readmitted within 30 days of THIS discharge
    inpatient["readmitted_30d"] = 0
    for patient_id, grp in inpatient.groupby("patient_id"):
        grp = grp.sort_values("start_date_key")
        idx = grp.index.to_list()
        starts, stops = grp["start_date_key"].values, grp["stop_date_key"].values
        for i in range(len(idx) - 1):
            gap_days = (starts[i + 1] - stops[i]) / pd.Timedelta(days=1)
            if 0 <= gap_days <= 30:
                inpatient.loc[idx[i], "readmitted_30d"] = 1

    # Feature: length of stay
    inpatient["length_of_stay_days"] = (
        (inpatient["stop_date_key"] - inpatient["start_date_key"]) / pd.Timedelta(days=1)
    ).clip(lower=0)

    # Feature: prior inpatient admissions in the preceding 365 days (known at discharge)
    inpatient["prior_admissions_365d"] = 0
    for patient_id, grp in inpatient.groupby("patient_id"):
        grp = grp.sort_values("start_date_key")
        starts = grp["start_date_key"].values
        idx = grp.index.to_list()
        for i in range(len(idx)):
            window_start = starts[i] - pd.Timedelta(days=365)
            prior_count = ((starts[:i] >= window_start) & (starts[:i] < starts[i])).sum()
            inpatient.loc[idx[i], "prior_admissions_365d"] = prior_count

    # Feature: active condition count at time of encounter
    conditions = pd.read_sql("""
        SELECT patient_id, encounter_id, code FROM fact_conditions
    """, conn)
    cond_counts = conditions.groupby("encounter_id").size().rename("condition_count")
    inpatient = inpatient.merge(cond_counts, on="encounter_id", how="left")
    inpatient["condition_count"] = inpatient["condition_count"].fillna(0)

    # Feature: active medication count at time of encounter
    meds = pd.read_sql("SELECT encounter_id FROM fact_medications", conn)
    med_counts = meds.groupby("encounter_id").size().rename("medication_count")
    inpatient = inpatient.merge(med_counts, on="encounter_id", how="left")
    inpatient["medication_count"] = inpatient["medication_count"].fillna(0)

    conn.close()

    # Patient-level features
    df = inpatient.merge(patients, on="patient_id", how="left")
    df["age_at_encounter"] = (
        (df["start_date_key"] - df["birthdate"]) / pd.Timedelta(days=365.25)
    ).round(1)
    df["gender"] = df["gender"].map({"M": 1, "F": 0}).fillna(-1)

    feature_cols = [
        "encounter_id", "patient_id", "age_at_encounter", "gender",
        "length_of_stay_days", "prior_admissions_365d", "condition_count",
        "medication_count", "total_claim_cost", "readmitted_30d",
    ]
    result = df[feature_cols].dropna(subset=["age_at_encounter"])
    result.to_csv(FEATURES_PATH, index=False)
    log.info(f"Built feature table: {len(result)} inpatient encounters, "
              f"{result['readmitted_30d'].mean()*100:.1f}% positive rate (readmitted within 30 days)")
    return result


if __name__ == "__main__":
    build_features()
