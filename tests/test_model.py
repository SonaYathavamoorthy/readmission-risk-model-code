"""
Tests for feature engineering correctness and model sanity checks.
"""
import sys
from pathlib import Path

import joblib
import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
FEATURES_PATH = ROOT / "data" / "features.csv"
MODEL_PATH = ROOT / "models" / "readmission_model.joblib"


@pytest.fixture(scope="module")
def features_df():
    return pd.read_csv(FEATURES_PATH)


def test_features_file_exists_and_nonempty(features_df):
    assert len(features_df) > 0


def test_no_negative_length_of_stay(features_df):
    assert (features_df["length_of_stay_days"] >= 0).all()


def test_label_is_binary(features_df):
    assert set(features_df["readmitted_30d"].unique()) <= {0, 1}


def test_positive_rate_matches_known_analysis(features_df):
    # Cross-check against the 17.1% readmission rate found in the
    # original EHR pipeline's analysis -- should match, since both
    # are computed from the same underlying warehouse.
    rate = features_df["readmitted_30d"].mean()
    assert 0.15 <= rate <= 0.19


def test_model_output_is_valid_probability():
    """Basic sanity check: predictions are valid probabilities regardless
    of input, across a range of prior-admission counts."""
    model = joblib.load(MODEL_PATH)
    for n in range(6):
        row = pd.DataFrame([{
            "age_at_encounter": 60, "gender": 0, "length_of_stay_days": 3,
            "prior_admissions_365d": n, "condition_count": 2,
            "medication_count": 3, "total_claim_cost": 8000,
        }])
        proba = model.predict_proba(row)[0, 1]
        assert 0.0 <= proba <= 1.0


def test_known_limitation_prior_admissions_not_monotonic():
    """
    Documents a real, discovered limitation rather than hiding it: this
    model's predicted risk slightly DECREASES as prior_admissions_365d
    increases (0 prior admissions -> 11.1% risk; 4+ -> 10.2% risk), which
    contradicts real clinical literature (more prior admissions should
    signal higher, not lower, readmission risk). This almost certainly
    reflects the small training set (356 encounters, only 61 positive
    cases) rather than a true pattern, and is called out explicitly in
    the model card rather than papered over. This test locks in the
    documented (if undesirable) current behavior so a future retrain
    that changes it is a visible, deliberate change -- not a silent one.
    """
    model = joblib.load(MODEL_PATH)
    base = pd.DataFrame([{
        "age_at_encounter": 60, "gender": 0, "length_of_stay_days": 3,
        "prior_admissions_365d": 0, "condition_count": 2,
        "medication_count": 3, "total_claim_cost": 8000,
    }])
    high_prior = base.copy()
    high_prior["prior_admissions_365d"] = 4

    low_score = model.predict_proba(base)[0, 1]
    high_score = model.predict_proba(high_prior)[0, 1]
    # Documenting the current (undesirable) behavior, not endorsing it --
    # see model card "Known Limitations" for the real-world implication.
    assert high_score <= low_score
