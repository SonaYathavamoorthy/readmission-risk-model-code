"""
Trains and evaluates two models (logistic regression, random forest) to
predict 30-day inpatient readmission, compares them, and saves the
better one for the prediction app.

Usage: python src/train.py
"""
import json
import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
FEATURES_PATH = ROOT / "data" / "features.csv"
MODEL_PATH = ROOT / "models" / "readmission_model.joblib"
SCALER_PATH = ROOT / "models" / "scaler.joblib"
METRICS_PATH = ROOT / "models" / "metrics.json"

FEATURE_COLS = ["age_at_encounter", "gender", "length_of_stay_days",
                 "prior_admissions_365d", "condition_count", "medication_count",
                 "total_claim_cost"]


def evaluate(name, model, X_test, y_test) -> dict:
    proba = model.predict_proba(X_test)[:, 1]
    preds = model.predict(X_test)
    auc = roc_auc_score(y_test, proba)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, preds, average="binary", zero_division=0
    )
    cm = confusion_matrix(y_test, preds).tolist()
    log.info(f"{name}: AUC={auc:.3f} | Precision={precision:.3f} | Recall={recall:.3f} | F1={f1:.3f}")
    return {"model": name, "auc": round(auc, 3), "precision": round(precision, 3),
            "recall": round(recall, 3), "f1": round(f1, 3), "confusion_matrix": cm}


def main():
    df = pd.read_csv(FEATURES_PATH)
    X = df[FEATURE_COLS]
    y = df["readmitted_30d"]

    log.info(f"Dataset: {len(df)} encounters, {y.sum()} positive ({y.mean()*100:.1f}%)")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    logreg = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    logreg.fit(X_train_scaled, y_train)
    logreg_metrics = evaluate("LogisticRegression", logreg, X_test_scaled, y_test)

    rf = RandomForestClassifier(
        n_estimators=200, max_depth=5, class_weight="balanced", random_state=42
    )
    rf.fit(X_train, y_train)  # tree models don't need scaling
    rf_metrics = evaluate("RandomForest", rf, X_test, y_test)

    # Pick the better model by AUC (more robust than accuracy on an imbalanced 17% positive class)
    if rf_metrics["auc"] >= logreg_metrics["auc"]:
        best_model, best_name = rf, "RandomForest"
        joblib.dump(best_model, MODEL_PATH)
        uses_scaler = False
    else:
        best_model, best_name = logreg, "LogisticRegression"
        joblib.dump(best_model, MODEL_PATH)
        joblib.dump(scaler, SCALER_PATH)
        uses_scaler = True

    log.info(f"Selected {best_name} as the final model (higher AUC).")

    # Feature importance (RandomForest) or coefficients (LogisticRegression)
    if best_name == "RandomForest":
        importance = dict(zip(FEATURE_COLS, best_model.feature_importances_.round(4).tolist()))
    else:
        importance = dict(zip(FEATURE_COLS, best_model.coef_[0].round(4).tolist()))
    importance = dict(sorted(importance.items(), key=lambda x: -abs(x[1])))
    log.info(f"Feature importance/coefficients: {importance}")

    results = {
        "selected_model": best_name,
        "uses_scaler": uses_scaler,
        "feature_cols": FEATURE_COLS,
        "logreg": logreg_metrics,
        "random_forest": rf_metrics,
        "feature_importance": importance,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "positive_rate": round(float(y.mean()), 3),
    }
    METRICS_PATH.write_text(json.dumps(results, indent=2))
    log.info(f"Saved model to {MODEL_PATH}, metrics to {METRICS_PATH}")


if __name__ == "__main__":
    main()
