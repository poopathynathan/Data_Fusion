import numpy as np
import torch
import torch.nn.functional as F
import joblib
import pandas as pd

from mbt_sensor_classifier import MultiSensorBottleneckTransformer

CLASS_NAMES = {0: "Drone", 1: "Bird", 2: "Clutter"}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_inference_data(data_path):
    """Load sensor features and labels from the generated NPZ or CSV dataset."""
    if data_path.endswith(".npz"):
        with np.load(data_path) as data:
            if all(name in data for name in ("radar", "df", "tv", "it", "pan", "labels")):
                features = np.concatenate(
                    [data[name] for name in ("radar", "df", "tv", "it", "pan")],
                    axis=1,
                )
                labels = data["labels"]
            elif "X" in data and "y" in data:
                features, labels = data["X"], data["y"]
            else:
                raise ValueError(
                    "NPZ must contain radar/df/tv/it/pan/labels or X/y arrays"
                )
    elif data_path.endswith(".csv"):
        frame = pd.read_csv(data_path)
        feature_columns = [
            column
            for prefix in ("radar_", "df_", "tv_", "it_", "pan_")
            for column in frame.columns
            if column.startswith(prefix)
        ]
        if len(feature_columns) != 54 or "target_class_id" not in frame:
            raise ValueError(
                "CSV must contain 54 sensor feature columns and target_class_id"
            )
        features = frame[feature_columns].to_numpy(dtype=np.float32)
        labels = frame["target_class_id"].to_numpy(dtype=np.int64)
    else:
        raise ValueError("Dataset must be .npz or .csv")

    if features.ndim != 2 or features.shape[1] != 54:
        raise ValueError(f"Expected 54 sensor features; got shape {features.shape}")
    if labels.ndim != 1 or len(labels) != len(features):
        raise ValueError("Dataset features and labels have incompatible shapes")
    return features, labels.astype(np.int64, copy=False)


def test_checkpoint(
    model_path="best_mbt_model.pth",
    data_path="sensor_data_calibrated_3classes_5000.npz",
    scaler_path="sensor_scaler.joblib",
):
    # 1. Load the model architecture
    model = MultiSensorBottleneckTransformer(
        sensor_dims=[12, 13, 9, 12, 8],
        d_model=128,
        num_bottlenecks=4,
        num_layers=2,
        num_heads=4,
        num_classes=3,
        dropout=0.0
    ).to(DEVICE)

    # 2. Load the trained weights (.pth file)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    print(f"[OK] Successfully loaded model weights from: {model_path}")

    features, labels = load_inference_data(data_path)
    scaler = joblib.load(scaler_path)
    features = scaler.transform(features)

    if len(features) < 5:
        raise ValueError(f"Need at least 5 samples for inference; found {len(features)}")
    indices = np.random.default_rng(42).choice(len(features), size=5, replace=False)

    print("\n--- SAMPLE PREDICTIONS ---")
    for i, idx in enumerate(indices, 1):
        sample = features[idx]
        ground_truth = int(labels[idx])

        # Slice into 5 sensor inputs
        s1 = torch.tensor(sample[0:12], dtype=torch.float32).unsqueeze(0).to(DEVICE)
        s2 = torch.tensor(sample[12:25], dtype=torch.float32).unsqueeze(0).to(DEVICE)
        s3 = torch.tensor(sample[25:34], dtype=torch.float32).unsqueeze(0).to(DEVICE)
        s4 = torch.tensor(sample[34:46], dtype=torch.float32).unsqueeze(0).to(DEVICE)
        s5 = torch.tensor(sample[46:54], dtype=torch.float32).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = model(s1, s2, s3, s4, s5)
            probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()
            pred_class = int(np.argmax(probs))

        status = "CORRECT" if pred_class == ground_truth else "WRONG"
        print(f"Sample #{i} [{status}]")
        print(f"  Ground Truth: {CLASS_NAMES[ground_truth]}")
        print(f"  Predicted   : {CLASS_NAMES[pred_class]} ({probs[pred_class]*100:.1f}% confidence)")
        print(f"  Probabilities -> Drone: {probs[0]*100:.1f}% | Bird: {probs[1]*100:.1f}% | Clutter: {probs[2]*100:.1f}%\n")

if __name__ == "__main__":
    test_checkpoint()