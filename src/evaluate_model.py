from __future__ import annotations

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from src.config import PROCESSED_DATA_DIR


MODEL_RESULTS_FILE = PROCESSED_DATA_DIR / "model_results.csv"


def load_results() -> pd.DataFrame:
    return pd.read_csv(MODEL_RESULTS_FILE)


def evaluate(results: pd.DataFrame) -> None:
    y_true = results["is_anomalous"]
    y_pred = results["predicted_anomaly"]

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, digits=3))


def show_prediction_breakdown(results: pd.DataFrame) -> None:
    print("\nTrue Positives (correctly flagged anomalies):")
    tp = results[(results["is_anomalous"] == 1) & (results["predicted_anomaly"] == 1)]
    print(tp[["batch_id", "anomaly_type", "anomaly_score"]])

    print("\nFalse Positives (normal batches incorrectly flagged):")
    fp = results[(results["is_anomalous"] == 0) & (results["predicted_anomaly"] == 1)]
    print(fp[["batch_id", "anomaly_type", "anomaly_score"]])

    print("\nFalse Negatives (anomalies missed by model):")
    fn = results[(results["is_anomalous"] == 1) & (results["predicted_anomaly"] == 0)]
    print(fn[["batch_id", "anomaly_type", "anomaly_score"]])


def show_top_ranked_batches(results: pd.DataFrame, top_n: int = 10) -> None:
    print(f"\nTop {top_n} most suspicious batches by anomaly score:")
    ranked = results.sort_values(by="anomaly_score", ascending=False).head(top_n)
    print(
        ranked[
            [
                "batch_id",
                "anomaly_type",
                "is_anomalous",
                "predicted_anomaly",
                "anomaly_score",
            ]
        ]
    )


def main() -> None:
    results = load_results()

    print("Loaded model results successfully.")
    evaluate(results)
    show_prediction_breakdown(results)
    show_top_ranked_batches(results, top_n=10)


if __name__ == "__main__":
    main()