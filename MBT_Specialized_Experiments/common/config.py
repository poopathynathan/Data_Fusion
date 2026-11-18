
from pathlib import Path
import torch

# Experiment directory
ROOT = Path(__file__).resolve().parent.parent

# Parent MBT directory
MODEL_ROOT = ROOT.parent

# Dataset
CSV_PATH = MODEL_ROOT / "sensor_data_calibrated_3classes_5000.csv"

# Trained model and scaler
MODEL_PATH = MODEL_ROOT / "mbt_specialized_3layers.pt"
SCALER_PATH = MODEL_ROOT / "mbt_specialized_3layers_scaler.joblib"

# Reproducibility
RANDOM_SEED = 42

# Evaluation settings
BATCH_SIZE = 64

# Temporary labels: replace with actual dataset class names
CLASS_NAMES = ["Class 0", "Class 1", "Class 2"]

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Results directory
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
