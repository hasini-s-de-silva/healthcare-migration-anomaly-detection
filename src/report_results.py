from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.config import FIGURES_DIR, PROCESSED_DATA_DIR


MODEL_RESULTS_FILE = PROCESSED_DATA_DIR / "model_results.csv"

FEATURE_COLUMNS = [
    "missing_dob_pct",
    "invalid_zip_pct",
    "duplicate_patient_id_pct",
    "orphan_encounter_pct",
    "orphan_condition_pct",
]


def load_results() -> pd.DataFrame:
    """
    Load model results table.
    """
    return pd.read_csv(MODEL_RESULTS_FILE)


def plot_anomaly_scores(results: pd.DataFrame) -> None:
    """
    Bar chart of anomaly scores by batch.
    """
    df = results.sort_values("anomaly_score", ascending=False).copy()

    plt.figure(figsize=(12, 6))
    plt.bar(df["batch_id"], df["anomaly_score"])
    plt.xticks(rotation=45)
    plt.xlabel("Batch ID")
    plt.ylabel("Anomaly Score")
    plt.title("Batch Anomaly Scores")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "anomaly_scores.png", dpi=300)
    plt.close()


def plot_feature_heatmap(results: pd.DataFrame) -> None:
    """
    Simple heatmap-like image using matplotlib only.
    Rows = batches, columns = quality features.
    """
    df = results.sort_values("anomaly_score", ascending=False).copy()
    heatmap_data = df[FEATURE_COLUMNS]

    plt.figure(figsize=(10, 8))
    plt.imshow(heatmap_data, aspect="auto")
    plt.colorbar(label="Feature Value")
    plt.xticks(range(len(FEATURE_COLUMNS)), FEATURE_COLUMNS, rotation=45, ha="right")
    plt.yticks(range(len(df)), df["batch_id"])
    plt.title("Batch Quality Feature Heatmap")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "feature_heatmap.png", dpi=300)
    plt.close()


def plot_normal_vs_anomalous(results: pd.DataFrame) -> None:
    """
    Compare average feature values for normal vs anomalous batches.
    """
    grouped = (
        results.groupby("is_anomalous")[FEATURE_COLUMNS]
        .mean()
        .rename(index={0: "normal", 1: "anomalous"})
    )

    plt.figure(figsize=(10, 6))
    x = range(len(FEATURE_COLUMNS))
    width = 0.35

    plt.bar([i - width / 2 for i in x], grouped.loc["normal"], width=width, label="Normal")
    plt.bar([i + width / 2 for i in x], grouped.loc["anomalous"], width=width, label="Anomalous")

    plt.xticks(x, FEATURE_COLUMNS, rotation=45, ha="right")
    plt.ylabel("Average Feature Value")
    plt.title("Normal vs Anomalous Batch Feature Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "normal_vs_anomalous_features.png", dpi=300)
    plt.close()


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading model results...")
    results = load_results()

    print("Generating anomaly score chart...")
    plot_anomaly_scores(results)

    print("Generating feature heatmap...")
    plot_feature_heatmap(results)

    print("Generating normal vs anomalous comparison chart...")
    plot_normal_vs_anomalous(results)

    print("\nSaved figures to:")
    print(FIGURES_DIR)
    print("\nFiles created:")
    print("- anomaly_scores.png")
    print("- feature_heatmap.png")
    print("- normal_vs_anomalous_features.png")


if __name__ == "__main__":
    main()