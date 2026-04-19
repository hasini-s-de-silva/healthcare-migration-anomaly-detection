from __future__ import annotations

import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from src.config import PROCESSED_DATA_DIR


MODEL_FEATURES_FILE = PROCESSED_DATA_DIR / "model_features.csv"
MODEL_RESULTS_FILE = PROCESSED_DATA_DIR / "model_results.csv"


FEATURE_COLUMNS = [
    "patient_rows",
    "encounter_rows",
    "condition_rows",
    "missing_dob_pct",
    "invalid_zip_pct",
    "duplicate_patient_id_pct",
    "orphan_encounter_pct",
    "orphan_condition_pct",
]


def load_data() -> pd.DataFrame:
    return pd.read_csv(MODEL_FEATURES_FILE)


def prepare_features(df: pd.DataFrame):
    X = df[FEATURE_COLUMNS].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled


def train_model(X):
    model = IsolationForest(
        n_estimators=200,
        contamination=0.25,
        random_state=42,
    )
    model.fit(X)
    return model


def score_results(df: pd.DataFrame, model, X):
    results = df.copy()

    # Higher score = more anomalous
    results["anomaly_score"] = -model.decision_function(X)

    # IsolationForest: -1 = anomaly, 1 = normal
    raw_pred = model.predict(X)
    results["predicted_anomaly"] = (raw_pred == -1).astype(int)

    return results


def main():
    print("Loading model feature data...")
    df = load_data()

    print("Preparing features...")
    X = prepare_features(df)

    print("Training Isolation Forest...")
    model = train_model(X)

    print("Scoring batches...")
    results = score_results(df, model, X)

    results = results.sort_values(by="anomaly_score", ascending=False)
    results.to_csv(MODEL_RESULTS_FILE, index=False)

    print("\nTop suspicious batches:")
    print(
        results[
            [
                "batch_id",
                "anomaly_type",
                "is_anomalous",
                "predicted_anomaly",
                "anomaly_score",
            ]
        ]
    )

    print(f"\nSaved results to: {MODEL_RESULTS_FILE}")


if __name__ == "__main__":
    main()