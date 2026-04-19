from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd

from src.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR


def extract_batch_ids() -> List[str]:
    """
    Find all batch IDs based on files in data/interim.

    Example expected files:
        batch_01_patients.csv
        batch_01_encounters.csv
        batch_01_conditions.csv
    """
    batch_ids = set()

    for file_path in INTERIM_DATA_DIR.glob("batch_*_patients.csv"):
        name = file_path.name.replace("_patients.csv", "")
        batch_ids.add(name)

    return sorted(batch_ids)


def load_batch_tables(batch_id: str) -> Dict[str, pd.DataFrame]:
    """
    Load the three batch tables for a given batch_id.
    """
    patients_path = INTERIM_DATA_DIR / f"{batch_id}_patients.csv"
    encounters_path = INTERIM_DATA_DIR / f"{batch_id}_encounters.csv"
    conditions_path = INTERIM_DATA_DIR / f"{batch_id}_conditions.csv"

    patients = pd.read_csv(patients_path)
    encounters = pd.read_csv(encounters_path)
    conditions = pd.read_csv(conditions_path)

    return {
        "patients": patients,
        "encounters": encounters,
        "conditions": conditions,
    }


def count_invalid_zip(patients: pd.DataFrame) -> int:
    """
    Count ZIP values that are not valid 5-digit strings.

    Important:
    ZIP codes should be treated as strings, not numeric values,
    because leading zeros may be lost when CSVs are read.
    """
    if "ZIP" not in patients.columns:
        return 0

    zip_series = patients["ZIP"].copy()

    # Handle missing values first
    missing_mask = zip_series.isna()

    # Convert to string safely
    zip_series = zip_series.astype(str).str.strip()

    # Remove trailing .0 if values came through as floats like 2139.0
    zip_series = zip_series.str.replace(r"\.0$", "", regex=True)

    # Left-pad numeric-looking ZIPs to 5 digits
    zip_series = zip_series.str.zfill(5)

    # Valid ZIP must now be exactly 5 digits
    valid_mask = zip_series.str.fullmatch(r"\d{5}")

    # Treat originally missing values as invalid too
    valid_mask = valid_mask & (~missing_mask)

    invalid_count = (~valid_mask).sum()
    return int(invalid_count)


def count_duplicate_patient_ids(patients: pd.DataFrame) -> int:
    """
    Count duplicated patient Id rows beyond the first occurrence.
    """
    if "Id" not in patients.columns:
        return 0

    return int(patients["Id"].duplicated().sum())


def count_orphan_encounters(encounters: pd.DataFrame, patients: pd.DataFrame) -> int:
    """
    Count encounter rows whose PATIENT value is not found in patients.Id.
    """
    if "PATIENT" not in encounters.columns or "Id" not in patients.columns:
        return 0

    patient_ids = set(patients["Id"].dropna())
    orphan_mask = ~encounters["PATIENT"].isin(patient_ids)

    return int(orphan_mask.sum())


def count_orphan_conditions(conditions: pd.DataFrame, patients: pd.DataFrame) -> int:
    """
    Count condition rows whose PATIENT value is not found in patients.Id.
    """
    if "PATIENT" not in conditions.columns or "Id" not in patients.columns:
        return 0

    patient_ids = set(patients["Id"].dropna())
    orphan_mask = ~conditions["PATIENT"].isin(patient_ids)

    return int(orphan_mask.sum())


def pct(numerator: int, denominator: int) -> float:
    """
    Safe percentage helper.
    """
    if denominator == 0:
        return 0.0
    return numerator / denominator


def validate_batch(batch_id: str) -> Dict[str, object]:
    """
    Run validation checks for a single batch and return summary metrics.
    """
    batch_tables = load_batch_tables(batch_id)

    patients = batch_tables["patients"]
    encounters = batch_tables["encounters"]
    conditions = batch_tables["conditions"]

    patient_rows = len(patients)
    encounter_rows = len(encounters)
    condition_rows = len(conditions)

    missing_dob_count = int(patients["BIRTHDATE"].isna().sum()) if "BIRTHDATE" in patients.columns else 0
    invalid_zip_count = count_invalid_zip(patients)
    duplicate_patient_id_count = count_duplicate_patient_ids(patients)

    orphan_encounter_count = count_orphan_encounters(encounters, patients)
    orphan_condition_count = count_orphan_conditions(conditions, patients)

    result = {
        "batch_id": batch_id,
        "patient_rows": patient_rows,
        "encounter_rows": encounter_rows,
        "condition_rows": condition_rows,
        "missing_dob_count": missing_dob_count,
        "missing_dob_pct": pct(missing_dob_count, patient_rows),
        "invalid_zip_count": invalid_zip_count,
        "invalid_zip_pct": pct(invalid_zip_count, patient_rows),
        "duplicate_patient_id_count": duplicate_patient_id_count,
        "duplicate_patient_id_pct": pct(duplicate_patient_id_count, patient_rows),
        "orphan_encounter_count": orphan_encounter_count,
        "orphan_encounter_pct": pct(orphan_encounter_count, encounter_rows),
        "orphan_condition_count": orphan_condition_count,
        "orphan_condition_pct": pct(orphan_condition_count, condition_rows),
    }

    return result


def main() -> None:
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    batch_ids = extract_batch_ids()
    if not batch_ids:
        raise FileNotFoundError(
            "No batch files found in data/interim/. "
            "Run `python -m src.simulate_batches` first."
        )

    print(f"Found {len(batch_ids)} batches.")
    validation_results = []

    for batch_id in batch_ids:
        result = validate_batch(batch_id)
        validation_results.append(result)

        print(
            f"{batch_id}: "
            f"patients={result['patient_rows']}, "
            f"encounters={result['encounter_rows']}, "
            f"conditions={result['condition_rows']}, "
            f"missing_dob_pct={result['missing_dob_pct']:.3f}, "
            f"invalid_zip_pct={result['invalid_zip_pct']:.3f}, "
            f"duplicate_patient_id_pct={result['duplicate_patient_id_pct']:.3f}, "
            f"orphan_encounter_pct={result['orphan_encounter_pct']:.3f}, "
            f"orphan_condition_pct={result['orphan_condition_pct']:.3f}"
        )

    validation_df = pd.DataFrame(validation_results)
    output_path = PROCESSED_DATA_DIR / "batch_validation_summary.csv"
    validation_df.to_csv(output_path, index=False)

    print("\nValidation summary saved to:")
    print(output_path)


if __name__ == "__main__":
    main()