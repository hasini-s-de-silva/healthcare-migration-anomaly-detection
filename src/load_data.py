from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd

from src.config import CORE_TABLES, PREVIEW_N_ROWS


def check_required_files() -> None:
    """
    Ensure the required raw Synthea CSV files exist.
    Raises FileNotFoundError if any are missing.
    """
    missing_files = [path for path in CORE_TABLES.values() if not path.exists()]

    if missing_files:
        missing_str = "\n".join(str(path) for path in missing_files)
        raise FileNotFoundError(
            f"Missing required raw data files:\n{missing_str}\n\n"
            "Please place the Synthea CSV exports into data/raw/."
        )


def load_csv(file_path: Path) -> pd.DataFrame:
    """
    Load a CSV file into a pandas DataFrame.
    """
    return pd.read_csv(file_path)


def load_core_tables() -> Dict[str, pd.DataFrame]:
    """
    Load the core Synthea tables defined in config.py.

    Returns:
        dict: Mapping of table name to DataFrame
    """
    check_required_files()

    tables = {}
    for table_name, file_path in CORE_TABLES.items():
        tables[table_name] = load_csv(file_path)

    return tables


def summarise_table(table_name: str, df: pd.DataFrame) -> None:
    """
    Print a quick summary of a loaded table.
    """
    print("=" * 80)
    print(f"TABLE: {table_name}")
    print("=" * 80)
    print(f"Shape: {df.shape}")
    print("\nColumns:")
    print(list(df.columns))

    print("\nData types:")
    print(df.dtypes)

    print(f"\nPreview (first {PREVIEW_N_ROWS} rows):")
    print(df.head(PREVIEW_N_ROWS))

    print("\nMissing values:")
    print(df.isnull().sum().sort_values(ascending=False).head(10))

    print("\n")


def main() -> None:
    """
    Load and inspect the core Synthea tables.
    """
    tables = load_core_tables()

    print("\nSuccessfully loaded core tables.\n")
    for table_name, df in tables.items():
        summarise_table(table_name, df)


if __name__ == "__main__":
    main()