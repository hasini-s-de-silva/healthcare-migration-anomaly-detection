from __future__ import annotations

import pandas as pd

from src.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR


VALIDATION_SUMMARY_FILE = PROCESSED_DATA_DIR / "batch_validation_summary.csv"
TRUTH_LOG_FILE = INTERIM_DATA_DIR / "batch_truth_log.csv"
MODEL_FEATURES_FILE = PROCESSED_DATA_DIR / "model_features.csv"


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


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load validation summary and truth log.
    """
    validation_df = pd.read_csv(VALIDATION_SUMMARY_FILE)
    truth_df = pd.read_csv(TRUTH_LOG_FILE)
    return validation_df, truth_df


def build_feature_table(
    validation_df: pd.DataFrame,
    truth_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge validation outputs with truth labels for evaluation.
    """
    merged = validation_df.merge(
        truth_df[["batch_id", "anomaly_type", "is_anomalous"]],
        on="batch_id",
        how="left",
    )

    # Ensure expected numeric feature columns are numeric
    for col in FEATURE_COLUMNS:
        merged[col] = pd.to_numeric(merged[col], errors="coerce")

    return merged


def main() -> None:
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    validation_df, truth_df = load_inputs()
    feature_df = build_feature_table(validation_df, truth_df)

    feature_df.to_csv(MODEL_FEATURES_FILE, index=False)

    print("Model feature table created successfully.")
    print(f"Saved to: {MODEL_FEATURES_FILE}")
    print("\nColumns:")
    print(list(feature_df.columns))
    print("\nPreview:")
    print(feature_df.head())


if __name__ == "__main__":
    main()