"""Project paths and reproducible training defaults."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DATA_PATH = DATA_DIR / "energydata_complete.csv"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
METRIC_DIR = OUTPUT_DIR / "metrics"
PREDICTION_DIR = OUTPUT_DIR / "predictions"
REPORT_DIR = OUTPUT_DIR / "reports"

TARGET = "Appliances"
SEED = 48
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_EPOCHS = 150
EARLY_STOPPING_PATIENCE = 15
HIDDEN_LAYERS = (128, 64, 32)
