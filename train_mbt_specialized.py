"""
Train the specialized multi-bottleneck MBT on the calibrated 5,000-row CSV.

Example:
    python train_mbt_specialized.py --layers 3

Try:
    python train_mbt_specialized.py --layers 2
    python train_mbt_specialized.py --layers 3
    python train_mbt_specialized.py --layers 4

Outputs are saved with layer-specific names. The original MBT checkpoint
and scaler are not overwritten.
"""

import argparse
import copy
import json
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from mbt_specialized import FEATURE_NAMES, SpecializedBottleneckTransformer


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="sensor_data_calibrated_3classes_5000.csv")
    parser.add_argument("--layers", type=int, default=3, choices=[2, 3, 4, 6])
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=7)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Could not find {csv_path}. Put the CSV beside this script "
            "or pass --csv with its path."
        )

    df = pd.read_csv(csv_path)
    required = FEATURE_NAMES + ["target_class_id"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    X = df[FEATURE_NAMES].to_numpy(dtype=np.float32)
    y = df["target_class_id"].to_numpy(dtype=np.int64)

    if not np.isfinite(X).all():
        raise ValueError("Feature matrix contains NaN or infinite values.")
    classes = np.unique(y)
    if len(classes) != 3:
        raise ValueError(f"Expected 3 classes, found {classes.tolist()}")
    if not np.array_equal(classes, np.array([0, 1, 2])):
        raise ValueError(
            "This script expects target_class_id values 0, 1, 2. "
            f"Found {classes.tolist()}."
        )

    # Reproduce the 70/15/15 stratified split used in the earlier baseline.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=args.seed, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=args.seed, stratify=y_temp
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, pin_memory=(device.type == "cuda")
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, pin_memory=(device.type == "cuda")
    )

    model = SpecializedBottleneckTransformer(
        num_classes=3,
        d_model=128,
        n_heads=4,
        num_layers=args.layers,
        ff_dim=512,
        dropout=0.1,
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_true, train_pred = [], []

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss_sum += loss.item() * xb.size(0)
            train_true.extend(yb.detach().cpu().tolist())
            train_pred.extend(logits.argmax(dim=1).detach().cpu().tolist())

        scheduler.step()
        train_loss = train_loss_sum / len(train_ds)
        train_acc = accuracy_score(train_true, train_pred)
        train_f1 = f1_score(train_true, train_pred, average="macro")

        model.eval()
        val_loss_sum = 0.0
        val_true, val_pred = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_loss_sum += loss.item() * xb.size(0)
                val_true.extend(yb.cpu().tolist())
                val_pred.extend(logits.argmax(dim=1).cpu().tolist())

        val_loss = val_loss_sum / len(val_ds)
        val_acc = accuracy_score(val_true, val_pred)
        val_f1 = f1_score(val_true, val_pred, average="macro")

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"train_f1={train_f1:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
            f"val_f1={val_f1:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping: no validation-loss improvement for {args.patience} epochs.")
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint.")

    model.load_state_dict(best_state)
    model.eval()

    all_probs, all_true = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            logits = model(xb.to(device))
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
            all_true.extend(yb.numpy().tolist())

    probabilities = np.concatenate(all_probs, axis=0)
    predictions = probabilities.argmax(axis=1)

    test_accuracy = accuracy_score(all_true, predictions)
    test_macro_f1 = f1_score(all_true, predictions, average="macro")
    test_log_loss = log_loss(all_true, probabilities, labels=[0, 1, 2])
    report = classification_report(
        all_true, predictions,
        labels=[0, 1, 2],
        target_names=["Drone", "Bird", "Clutter"],
        digits=4,
        zero_division=0,
    )
    cm = confusion_matrix(all_true, predictions, labels=[0, 1, 2])

    stem = f"mbt_specialized_{args.layers}layers"
    checkpoint_path = Path(f"{stem}.pt")
    scaler_path = Path(f"{stem}_scaler.joblib")
    metrics_path = Path(f"{stem}_metrics.json")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "num_classes": 3,
            "d_model": 128,
            "n_heads": 4,
            "num_layers": args.layers,
            "ff_dim": 512,
            "dropout": 0.1,
            "feature_names": FEATURE_NAMES,
            "class_names": ["Drone", "Bird", "Clutter"],
            "routing_description": "Domain-guided overlapping feature-to-bottleneck routing",
            "seed": args.seed,
        },
        checkpoint_path,
    )
    joblib.dump(scaler, scaler_path)

    metrics = {
        "layers": args.layers,
        "seed": args.seed,
        "train_samples": len(train_ds),
        "validation_samples": len(val_ds),
        "test_samples": len(test_ds),
        "test_accuracy": float(test_accuracy),
        "test_macro_f1": float(test_macro_f1),
        "test_log_loss": float(test_log_loss),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("\n===== HELD-OUT TEST RESULTS =====")
    print(f"Accuracy : {test_accuracy:.4f}")
    print(f"Macro F1 : {test_macro_f1:.4f}")
    print(f"Log loss : {test_log_loss:.4f}")
    print("\nConfusion matrix (Drone, Bird, Clutter):")
    print(cm)
    print("\nClassification report:")
    print(report)
    print("\nSaved files:")
    print(f"  Model : {checkpoint_path}")
    print(f"  Scaler: {scaler_path}")
    print(f"  Metrics: {metrics_path}")


if __name__ == "__main__":
    main()
