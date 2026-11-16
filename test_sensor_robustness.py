import numpy as np
import torch
import torch.nn.functional as F
import joblib
from mbt_sensor_classifier import MultiSensorBottleneckTransformer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_scenario(model, X_scaled, y, name, active_sensors=[True, True, True, True, True]):
    """
    Evaluates model accuracy when certain sensors are forced to zero (simulating offline/jammed sensors).
    active_sensors: [S1_Radar, S2_RF, S3_TV, S4_IT, S5_PAN]
    """
    correct = 0
    total = len(y)

    for i in range(total):
        sample = X_scaled[i].copy()

        # Extract features per sensor
        s1_raw = sample[0:12]  if active_sensors[0] else np.zeros(12)
        s2_raw = sample[12:25] if active_sensors[1] else np.zeros(13)
        s3_raw = sample[25:34] if active_sensors[2] else np.zeros(9)
        s4_raw = sample[34:46] if active_sensors[3] else np.zeros(12)
        s5_raw = sample[46:54] if active_sensors[4] else np.zeros(8)

        s1 = torch.tensor(s1_raw, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        s2 = torch.tensor(s2_raw, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        s3 = torch.tensor(s3_raw, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        s4 = torch.tensor(s4_raw, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        s5 = torch.tensor(s5_raw, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = model(s1, s2, s3, s4, s5)
            pred = logits.argmax(dim=1).item()

        if pred == y[i]:
            correct += 1

    acc = (correct / total) * 100
    print(f"  {name:<42} | Accuracy: {acc:6.2f}% ({correct}/{total})")
    return acc

def run_stress_test(model_path="best_mbt_model.pth", data_path="synthetic_sensor_data_3classes_5000.npz"):
    # 1. Load Model
    model = MultiSensorBottleneckTransformer(
        sensor_dims=[12, 13, 9, 12, 8],
        d_model=128,
        num_bottlenecks=4,
        num_layers=2,
        num_heads=4,
        num_classes=3,
        dropout=0.0
    ).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    # 2. Load & Scale Data (Testing on last 750 samples as held-out test set)
    data = np.load(data_path)
    X, y = data["X"][-750:], data["y"][-750:]

    try:
        scaler = joblib.load("sensor_scaler.joblib")
        X = scaler.transform(X)
    except:
        pass

    print("==========================================================================")
    print("                  MULTI-SENSOR STRESS TEST & ROBUSTNESS                   ")
    print("==========================================================================")
    
    # Baseline
    evaluate_scenario(model, X, y, "1. Full Suite (All 5 Sensors Active)", [True, True, True, True, True])
    
    # Individual sensor failures
    evaluate_scenario(model, X, y, "2. Radio-Silent Target (No RF Scanner)", [True, False, True, True, True])
    evaluate_scenario(model, X, y, "3. Night / Fog (No Daylight TV Camera)", [True, True, False, True, True])
    evaluate_scenario(model, X, y, "4. Thermal Failure (No IR Camera)",      [True, True, True, False, True])
    evaluate_scenario(model, X, y, "5. Radar Jammed / Occluded (No Radar)",  [False, True, True, True, True])
    
    # Severe multi-sensor outages
    print("--------------------------------------------------------------------------")
    evaluate_scenario(model, X, y, "6. EO/IR Only (Radar & RF Jammed)",      [False, False, True, True, True])
    evaluate_scenario(model, X, y, "7. Radar Only (No Optical, No RF)",      [True, False, False, False, False])
    evaluate_scenario(model, X, y, "8. RF Only (Beyond Visual/Radar Range)", [False, True, False, False, False])
    print("==========================================================================")

if __name__ == "__main__":
    run_stress_test()