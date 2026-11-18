import numpy as np
import pandas as pd
from mbt_specialized import FEATURE_NAMES
from common.config import RANDOM_SEED, RESULTS_DIR
from common.data import load_test_data
from common.model import load_trained_model, verify_weights_unchanged
from common.evaluate import evaluate

def main():
    X, y = load_test_data()
    model, snapshot = load_trained_model()
    rng = np.random.default_rng(RANDOM_SEED)

    baseline, _ = evaluate(model, X, y, "E2_reference_baseline")
    rows = []
    for j, feature_name in enumerate(FEATURE_NAMES):
        changed = X.copy()
        changed[:, j] = X[rng.permutation(len(X)), j]
        metrics, _ = evaluate(model, changed, y, f"E2_feature_{j:02d}")
        rows.append({
            "Feature_Index": j,
            "Feature_Name": feature_name,
            **metrics,
            "Accuracy_Drop": baseline["Accuracy"] - metrics["Accuracy"],
            "Macro_F1_Drop": baseline["Macro_F1"] - metrics["Macro_F1"]
        })

    pd.DataFrame(rows).sort_values(
        "Macro_F1_Drop", ascending=False
    ).to_csv(f"{RESULTS_DIR}/E2_feature_importance.csv", index=False)
    verify_weights_unchanged(model, snapshot)

if __name__ == "__main__":
    main()
