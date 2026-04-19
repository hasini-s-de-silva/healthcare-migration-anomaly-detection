from pathlib import Path

# Project root = parent of /src
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data folders
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Other folders
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
REPORTS_DIR = OUTPUTS_DIR / "reports"
FIGURES_DIR = OUTPUTS_DIR / "figures"

# Expected Synthea files
PATIENTS_FILE = RAW_DATA_DIR / "patients.csv"
ENCOUNTERS_FILE = RAW_DATA_DIR / "encounters.csv"
CONDITIONS_FILE = RAW_DATA_DIR / "conditions.csv"

# Tables we care about for version 1
CORE_TABLES = {
    "patients": PATIENTS_FILE,
    "encounters": ENCOUNTERS_FILE,
    "conditions": CONDITIONS_FILE,
}

# Number of preview rows to print when inspecting data
PREVIEW_N_ROWS = 5