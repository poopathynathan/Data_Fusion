"""Train and evaluate the sensor bottleneck transformer."""

import os

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from mbt_sensor_classifier import (
    SENSOR_DIMS,
    MultiSensorBottleneckTransformer,
    SensorFusionDataset,
)


CLASS_NAMES = ["Drone (0)", "Bird (1)", "Clutter (2)"]
DEFAULT_NPZ = "sensor_data_calibrated_3classes_5000.npz"
DEFAULT_CSV = "sensor_data_calibrated_3classes_5000.csv"


def load_dataset(file_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load sensor columns in model order and their integer class labels."""
    if file_path.endswith(".npz"):
        with np.load(file_path) as data:
            sensor_arrays = [data[name] for name in ("radar", "df", "tv", "it", "pan")]
            features = np.concatenate(sensor_arrays, axis=1)
            labels = data["labels"]
    elif file_path.endswith(".csv"):
        frame = pd.read_csv(file_path)
        sensor_columns = [
            column
            for prefix in ("radar_", "df_", "tv_", "it_", "pan_")
            for column in frame.columns
            if column.startswith(prefix)
        ]
        if len(sensor_columns) != sum(SENSOR_DIMS):
            raise ValueError(
                f"CSV must contain {sum(SENSOR_DIMS)} sensor features; "
                f"found {len(sensor_columns)}"
            )
        label_column = "target_class_id" if "target_class_id" in frame else None
        if label_column is None:
            raise ValueError("CSV must contain a target_class_id column")
        features = frame[sensor_columns].to_numpy(dtype=np.float32)
        labels = frame[label_column].to_numpy(dtype=np.int64)
    else:
        raise ValueError("File must be .npz or .csv")

    expected_width = sum(SENSOR_DIMS)
    if features.ndim != 2 or features.shape[1] != expected_width:
        raise ValueError(
            f"Expected {expected_width} sensor features; received {features.shape}"
        )
    if labels.ndim != 1 or len(labels) != len(features):
        raise ValueError("Dataset features and labels have incompatible shapes")
    return features, labels.astype(np.int64, copy=False)


def prepare_splits(file_path: str = DEFAULT_NPZ, seed: int = 42):
    """Create stratified 70/15/15 splits and scale using train data only."""
    features, labels = load_dataset(file_path)
    x_train, x_temp, y_train, y_temp = train_test_split(
        features, labels, test_size=0.30, random_state=seed, stratify=labels
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp, y_temp, test_size=0.50, random_state=seed, stratify=y_temp
    )

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)
    x_test = scaler.transform(x_test)
    joblib.dump(scaler, "sensor_scaler.joblib")
    print("[OK] Scaler saved as 'sensor_scaler.joblib'")

    train_ds = SensorFusionDataset(x_train, y_train)
    val_ds = SensorFusionDataset(x_val, y_val)
    test_ds = SensorFusionDataset(x_test, y_test)
    print(
        f"Data Split: Train={len(train_ds)}, Val={len(val_ds)}, "
        f"Test={len(test_ds)}"
    )
    return train_ds, val_ds, test_ds


def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_path = DEFAULT_NPZ if os.path.exists(DEFAULT_NPZ) else DEFAULT_CSV
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Dataset not found. Expected '{DEFAULT_NPZ}' or '{DEFAULT_CSV}'."
        )

    train_ds, val_ds, test_ds = prepare_splits(data_path)
    generator = torch.Generator().manual_seed(42)
    train_loader = DataLoader(
        train_ds, batch_size=64, shuffle=True, generator=generator
    )
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    model = MultiSensorBottleneckTransformer(
        sensor_dims=list(SENSOR_DIMS),
        d_model=128,
        num_bottlenecks=4,
        num_layers=2,
        num_heads=4,
        num_classes=3,
        dropout=0.1,
    ).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=3e-4, weight_decay=1e-4
    )
    epochs = 25
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    )
    checkpoint_path = "best_mbt_model.pth"
    best_val_loss = float("inf")

    print("\nStarting Training...")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            *sensor_inputs, labels = batch
            sensor_inputs = [sensor.to(device) for sensor in sensor_inputs]
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(*sensor_inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * labels.size(0)
            train_correct += (logits.argmax(dim=1) == labels).sum().item()
            train_total += labels.size(0)

        scheduler.step()
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch in val_loader:
                *sensor_inputs, labels = batch
                sensor_inputs = [sensor.to(device) for sensor in sensor_inputs]
                labels = labels.to(device)
                logits = model(*sensor_inputs)
                loss = criterion(logits, labels)
                val_loss += loss.item() * labels.size(0)
                val_correct += (logits.argmax(dim=1) == labels).sum().item()
                val_total += labels.size(0)

        epoch_val_loss = val_loss / val_total
        print(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"Train Loss: {train_loss / train_total:.4f} "
            f"Acc: {100 * train_correct / train_total:.2f}% | "
            f"Val Loss: {epoch_val_loss:.4f} "
            f"Acc: {100 * val_correct / val_total:.2f}%"
        )
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), checkpoint_path)

    print(f"\n[OK] Training complete. Best checkpoint: '{checkpoint_path}'")
    print("\nRunning Evaluation on Held-Out Test Set...")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in test_loader:
            *sensor_inputs, labels = batch
            sensor_inputs = [sensor.to(device) for sensor in sensor_inputs]
            logits = model(*sensor_inputs)
            all_preds.extend(logits.argmax(dim=1).cpu().numpy())
            all_labels.extend(labels.numpy())

    print("\n==================== TEST SET PERFORMANCE ====================")
    print(
        classification_report(
            all_labels, all_preds, labels=[0, 1, 2],
            target_names=CLASS_NAMES, digits=4, zero_division=0
        )
    )
    print("Confusion Matrix:")
    print("Pred ->      Drone   Bird   Clutter")
    matrix = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2])
    for index, row in enumerate(matrix):
        print(
            f"Actual {CLASS_NAMES[index]:<10}: "
            f"{row[0]:<7} {row[1]:<7} {row[2]:<7}"
        )


if __name__ == "__main__":
    train_model()
