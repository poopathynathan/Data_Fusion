import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from .config import DEVICE, BATCH_SIZE, CLASS_NAMES, RESULTS_DIR

def evaluate(model, X, y, experiment_name):
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    if X.ndim != 2 or X.shape[1] != 54:
        raise ValueError(f"Expected [N,54] input; received {X.shape}")

    loader = DataLoader(
        TensorDataset(torch.from_numpy(X), torch.from_numpy(y)),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    pred_all, true_all = [], []
    with torch.inference_mode():
        for xb, yb in loader:
            logits = model(xb.to(DEVICE))
            pred = logits.argmax(dim=1)
            pred_all.extend(pred.cpu().numpy().tolist())
            true_all.extend(yb.numpy().tolist())

    metrics = {
        "Experiment": experiment_name,
        "Accuracy": float(accuracy_score(true_all, pred_all)),
        "Macro_Precision": float(precision_score(true_all, pred_all, average="macro", zero_division=0)),
        "Macro_Recall": float(recall_score(true_all, pred_all, average="macro", zero_division=0)),
        "Macro_F1": float(f1_score(true_all, pred_all, average="macro", zero_division=0))
    }

    out = os.path.join(RESULTS_DIR, experiment_name)
    os.makedirs(out, exist_ok=True)
    report = classification_report(
        true_all, pred_all, labels=[0, 1, 2],
        target_names=CLASS_NAMES, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(true_all, pred_all, labels=[0, 1, 2])

    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
    with open(os.path.join(out, "classification_report.json"), "w") as f:
        json.dump(report, f, indent=4)
    np.savetxt(os.path.join(out, "confusion_matrix.csv"), cm, delimiter=",", fmt="%d")

    print(f"\n{experiment_name}:")
    for key, value in metrics.items():
        if key != "Experiment":
            print(f"{key}: {value:.4f}")
    print("Confusion matrix:\n", cm)
    return metrics, np.asarray(pred_all)
