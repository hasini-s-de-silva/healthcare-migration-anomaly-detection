from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.config import (
    INTERIM_DATA_DIR,
    PATIENTS_FILE,
    ENCOUNTERS_FILE,
    CONDITIONS_FILE,
)

# -----------------------------
# Configuration
# -----------------------------
N_BATCHES = 20
RANDOM_SEED = 42

ANOMALY_PLAN = {
    "batch_16": "missing_dob_spike",
    "batch_17": "invalid_zip_spike",
    "batch_18": "duplicate_patient_ids",
    "batch_19": "orphan_encounters",
    "batch_20": "orphan_conditions",
}


# -----------------------------
# Load source data
# -----------------------------
def load_source_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    patients = pd.read_csv(PATIENTS_FILE)
    encounters = pd.read_csv(ENCOUNTERS_FILE)
    conditions = pd.read_csv(CONDITIONS_FILE)
    return patients, encounters, conditions


# -----------------------------
# Batch creation
# -----------------------------
def assign_patient_batches(patients: pd.DataFrame, n_batches: int, seed: int) -> pd.DataFrame:
    """
    Shuffle patients and assign each patient to one batch.
    """
    patients = patients.sample(frac=1, random_state=seed).reset_index(drop=True)

    batch_labels = np.array(
        [f"batch_{i:02d}" for i in range(1, n_batches + 1)]
    )
    repeated_labels = np.resize(batch_labels, len(patients))
    patients["batch_id"] = repeated_labels

    return patients


def build_clean_batches(
    patients: pd.DataFrame,
    encounters: pd.DataFrame,
    conditions: pd.DataFrame,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Create clean patient-anchored batches.

    Returns:
        {
            "batch_01": {
                "patients": ...,
                "encounters": ...,
                "conditions": ...
            },
            ...
        }
    """
    batches: Dict[str, Dict[str, pd.DataFrame]] = {}

    for batch_id, batch_patients in patients.groupby("batch_id"):
        patient_ids = set(batch_patients["Id"])

        batch_encounters = encounters[encounters["PATIENT"].isin(patient_ids)].copy()

        encounter_ids = set(batch_encounters["Id"]) if "Id" in batch_encounters.columns else set()

        batch_conditions = conditions[conditions["PATIENT"].isin(patient_ids)].copy()

        # Keep only conditions tied to the batch's encounters when possible.
        # If ENCOUNTER is missing or not present in encounter_ids, we still keep by patient anchor.
        if "ENCOUNTER" in batch_conditions.columns and encounter_ids:
            # Keep rows with encounter IDs in this batch OR rows with null ENCOUNTER
            batch_conditions = batch_conditions[
                batch_conditions["ENCOUNTER"].isin(encounter_ids)
                | batch_conditions["ENCOUNTER"].isna()
            ].copy()

        batches[batch_id] = {
            "patients": batch_patients.drop(columns=["batch_id"]).copy(),
            "encounters": batch_encounters,
            "conditions": batch_conditions,
        }

    return batches


# -----------------------------
# Anomaly injection helpers
# -----------------------------
def inject_missing_dob_spike(df: pd.DataFrame, seed: int, frac: float = 0.20) -> pd.DataFrame:
    df = df.copy()
    rng = np.random.default_rng(seed)

    eligible_idx = df.index[df["BIRTHDATE"].notna()]
    n = max(1, int(len(eligible_idx) * frac)) if len(eligible_idx) > 0 else 0

    if n > 0:
        chosen = rng.choice(eligible_idx, size=n, replace=False)
        df.loc[chosen, "BIRTHDATE"] = np.nan

    return df


def inject_invalid_zip_spike(df: pd.DataFrame, seed: int, frac: float = 0.15) -> pd.DataFrame:
    df = df.copy()
    rng = np.random.default_rng(seed)

    if "ZIP" not in df.columns or len(df) == 0:
        return df

    n = max(1, int(len(df) * frac))
    chosen = rng.choice(df.index, size=n, replace=False)

    bad_values = ["ABC", "12X45", "999999999", "", "000", "INVALID"]
    replacements = rng.choice(bad_values, size=n, replace=True)

    df.loc[chosen, "ZIP"] = replacements
    return df


def inject_duplicate_patient_ids(df: pd.DataFrame, seed: int, frac: float = 0.10) -> pd.DataFrame:
    """
    Duplicate a subset of patient rows without changing Id,
    creating duplicated patient identifiers within the batch.
    """
    df = df.copy()
    rng = np.random.default_rng(seed)

    if len(df) == 0:
        return df

    n = max(1, int(len(df) * frac))
    chosen = rng.choice(df.index, size=n, replace=False)
    duplicated_rows = df.loc[chosen].copy()

    df = pd.concat([df, duplicated_rows], ignore_index=True)
    return df


def inject_orphan_encounters(df: pd.DataFrame, seed: int, frac: float = 0.10) -> pd.DataFrame:
    """
    Corrupt some encounter PATIENT references so they no longer map to a real patient.
    """
    df = df.copy()
    rng = np.random.default_rng(seed)

    if len(df) == 0 or "PATIENT" not in df.columns:
        return df

    n = max(1, int(len(df) * frac))
    chosen = rng.choice(df.index, size=n, replace=False)

    fake_ids = [f"orphan-patient-{i}" for i in range(n)]
    df.loc[chosen, "PATIENT"] = fake_ids

    return df


def inject_orphan_conditions(df: pd.DataFrame, seed: int, frac: float = 0.10) -> pd.DataFrame:
    """
    Corrupt some condition PATIENT references so they no longer map to a real patient.
    """
    df = df.copy()
    rng = np.random.default_rng(seed)

    if len(df) == 0 or "PATIENT" not in df.columns:
        return df

    n = max(1, int(len(df) * frac))
    chosen = rng.choice(df.index, size=n, replace=False)

    fake_ids = [f"orphan-condition-patient-{i}" for i in range(n)]
    df.loc[chosen, "PATIENT"] = fake_ids

    return df


# -----------------------------
# Apply anomalies
# -----------------------------
def apply_batch_anomaly(
    batch_id: str,
    batch_data: Dict[str, pd.DataFrame],
    seed: int,
) -> Tuple[Dict[str, pd.DataFrame], str]:
    """
    Apply a predefined anomaly to a given batch.
    """
    anomaly_type = ANOMALY_PLAN.get(batch_id, "none")

    patients = batch_data["patients"].copy()
    encounters = batch_data["encounters"].copy()
    conditions = batch_data["conditions"].copy()

    if anomaly_type == "missing_dob_spike":
        patients = inject_missing_dob_spike(patients, seed=seed)

    elif anomaly_type == "invalid_zip_spike":
        patients = inject_invalid_zip_spike(patients, seed=seed)

    elif anomaly_type == "duplicate_patient_ids":
        patients = inject_duplicate_patient_ids(patients, seed=seed)

    elif anomaly_type == "orphan_encounters":
        encounters = inject_orphan_encounters(encounters, seed=seed)

    elif anomaly_type == "orphan_conditions":
        conditions = inject_orphan_conditions(conditions, seed=seed)

    batch_data_out = {
        "patients": patients,
        "encounters": encounters,
        "conditions": conditions,
    }

    return batch_data_out, anomaly_type


# -----------------------------
# Save outputs
# -----------------------------
def save_batch_files(batch_id: str, batch_data: Dict[str, pd.DataFrame]) -> None:
    INTERIM_DATA_DIR.mkdir(parents=True, exist_ok=True)

    batch_data["patients"].to_csv(INTERIM_DATA_DIR / f"{batch_id}_patients.csv", index=False)
    batch_data["encounters"].to_csv(INTERIM_DATA_DIR / f"{batch_id}_encounters.csv", index=False)
    batch_data["conditions"].to_csv(INTERIM_DATA_DIR / f"{batch_id}_conditions.csv", index=False)


def save_truth_log(records: List[dict]) -> None:
    truth_log = pd.DataFrame(records)
    truth_log.to_csv(INTERIM_DATA_DIR / "batch_truth_log.csv", index=False)


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    print("Loading source data...")
    patients, encounters, conditions = load_source_data()

    print(f"Patients shape:   {patients.shape}")
    print(f"Encounters shape: {encounters.shape}")
    print(f"Conditions shape: {conditions.shape}")

    print("\nAssigning patients to batches...")
    patients_batched = assign_patient_batches(
        patients=patients,
        n_batches=N_BATCHES,
        seed=RANDOM_SEED,
    )

    print("Building clean patient-anchored batches...")
    clean_batches = build_clean_batches(
        patients=patients_batched,
        encounters=encounters,
        conditions=conditions,
    )

    truth_records = []

    print("Applying anomalies and saving batch files...")
    for i, batch_id in enumerate(sorted(clean_batches.keys()), start=1):
        batch_seed = RANDOM_SEED + i
        batch_data = clean_batches[batch_id]

        batch_data_out, anomaly_type = apply_batch_anomaly(
            batch_id=batch_id,
            batch_data=batch_data,
            seed=batch_seed,
        )

        save_batch_files(batch_id, batch_data_out)

        truth_records.append(
            {
                "batch_id": batch_id,
                "anomaly_type": anomaly_type,
                "is_anomalous": int(anomaly_type != "none"),
                "patient_rows": len(batch_data_out["patients"]),
                "encounter_rows": len(batch_data_out["encounters"]),
                "condition_rows": len(batch_data_out["conditions"]),
            }
        )

        print(
            f"{batch_id}: anomaly={anomaly_type}, "
            f"patients={len(batch_data_out['patients'])}, "
            f"encounters={len(batch_data_out['encounters'])}, "
            f"conditions={len(batch_data_out['conditions'])}"
        )

    save_truth_log(truth_records)

    print("\nDone.")
    print(f"Batch files written to: {INTERIM_DATA_DIR}")
    print(f"Truth log written to:   {INTERIM_DATA_DIR / 'batch_truth_log.csv'}")


if __name__ == "__main__":
    main()