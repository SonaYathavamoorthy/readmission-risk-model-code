# Model Card: 30-Day Readmission Risk Predictor

## Overview
Predicts the probability that an inpatient encounter will be followed
by another inpatient admission within 30 days, using features knowable
at discharge time. Trained on 356 inpatient encounters from a synthetic
EHR warehouse (see the companion [Synthetic EHR Patient Journey
Pipeline](../synthetic-ehr-pipeline) project).

## Intended use
Portfolio/demonstration project. **Not validated for clinical use.**
Illustrates the modeling workflow (feature engineering → train/compare
models → evaluate → deploy) a real health data science team would follow.

## Data
- Source: Synthea-generated synthetic patients (10-year histories, Virginia)
- 356 inpatient encounters, 61 positive labels (17.1% readmitted within 30 days)
- Features: age, gender, length of stay, prior admissions (365d),
  active condition count, active medication count, total claim cost

## Models compared

| Model | AUC | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | 0.793 | 0.350 | 0.933 | 0.509 |
| Random Forest (selected) | 0.803 | 0.344 | 0.733 | 0.468 |

Random Forest was selected on AUC. Logistic Regression actually has
higher recall — worth noting, since in a real discharge-planning
context, missing a high-risk patient (false negative) is usually
costlier than an unnecessary follow-up call (false positive). **A real
deployment would likely prefer the logistic regression model, or a
probability threshold tuned for recall, not the AUC-optimal choice.**
This is a deliberate simplification for portfolio scope, called out
rather than hidden.

## Known limitations

1. **Small dataset.** 356 encounters (61 positive) is far below what a
   production model would need. Metrics should be read as directional,
   not precise.
2. **Non-monotonic prior-admissions behavior (found via testing, not
   assumed).** The model's predicted risk slightly *decreases* as
   `prior_admissions_365d` increases (0 prior admissions → 11.1% risk;
   4+ → 10.2% risk) — the opposite of established clinical literature,
   where more prior admissions predicts higher, not lower, readmission
   risk. `tests/test_model.py` documents this explicitly rather than
   silently accepting it. It almost certainly reflects noise from the
   small sample rather than a real pattern, and any real deployment
   would need to address this before trusting the feature at all —
   e.g., more data, a monotonicity constraint, or dropping the feature.
3. **Synthetic data.** Synthea models disease progression realistically
   but doesn't capture the full complexity or noise of real EHR data
   (documentation errors, coding inconsistencies, care-setting variation).
4. **No external validation.** The model has only been tested on a
   held-out split of the same synthetic population — not on a different
   cohort, time period, or care setting.

## What would change this for production use
More data (thousands, not hundreds, of encounters), external validation
on a separate cohort, a recall-oriented threshold or cost-sensitive
loss function, monotonicity constraints on clinically-known-directional
features, and clinical review of the feature set with an actual care
team before any deployment near real decisions.
