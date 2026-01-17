from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

# data paths
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# models paths
MODELS_DIR = BASE_DIR / "models"

# reports paths
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# outputs/logs paths
OUTPUTS_DIR = BASE_DIR / "outputs"

# configs paths
CONFIGS_DIR = BASE_DIR / "configs"
