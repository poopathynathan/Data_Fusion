
import sys
from pathlib import Path

# Locate parent MBT directory
MODEL_ROOT = Path(__file__).resolve().parents[2]

if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from mbt_specialized import FEATURE_NAMES

from .config import CSV_PATH, SCALER_PATH, RANDOM_SEED


def load_test_data():

    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {CSV_PATH}")

    if not SCALER_PATH.exists():
        raise FileNotFoundError(f"Scaler not found: {SCALER_PATH}")

    df = pd.read_csv(CSV_PATH)

    required = list(FEATURE_NAMES) + ["target_class_id"]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    X = df[list(FEATURE_NAMES)].to_numpy(dtype=np.float32)
    y = df["target_class_id"].to_numpy(dtype=np.int64)

    if not np.isfinite(X).all():
        raise ValueError("Dataset contains NaN or infinite values.")

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=0.30,
        random_state=RANDOM_SEED,
        stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.50,
        random_state=RANDOM_SEED,
        stratify=y_temp
    )

    scaler = joblib.load(SCALER_PATH)
    X_test = scaler.transform(X_test).astype(np.float32)

    print(f"Dataset: {CSV_PATH}")
    print(f"Test samples: {len(X_test)}")
    print(f"Feature count: {X_test.shape[1]}")
    print("Class distribution:", dict(zip(*np.unique(y_test, return_counts=True))))

    return X_test, y_test
