from mbt_specialized import FEATURE_NAMES
from common.data import load_test_data
from common.model import load_trained_model, verify_weights_unchanged
from common.evaluate import evaluate

def main():
    X, y = load_test_data()
    model, snapshot = load_trained_model()

    prefixes = ["radar_", "df_", "tv_", "it_", "pan_"]
    for prefix in prefixes:
        idx = [i for i, name in enumerate(FEATURE_NAMES) if name.startswith(prefix)]
        changed = X.copy()
        # Zero standardized values: replace each feature with its training mean.
        changed[:, idx] = 0.0
        evaluate(model, changed, y, f"E3_ablate_{prefix[:-1].upper()}")

    verify_weights_unchanged(model, snapshot)

if __name__ == "__main__":
    main()
