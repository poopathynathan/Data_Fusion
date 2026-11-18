import numpy as np
from mbt_specialized import FEATURE_NAMES
from common.config import RANDOM_SEED
from common.data import load_test_data
from common.model import load_trained_model, verify_weights_unchanged
from common.evaluate import evaluate

def main():
    X, y = load_test_data()
    model, snapshot = load_trained_model()
    rng = np.random.default_rng(RANDOM_SEED)

    for prefix in ["radar_", "df_", "tv_", "it_", "pan_"]:
        idx = [i for i, name in enumerate(FEATURE_NAMES) if name.startswith(prefix)]
        permuted_idx = rng.permutation(idx)
        changed = X.copy()
        changed[:, idx] = X[:, permuted_idx]
        evaluate(model, changed, y, f"E4_reorder_{prefix[:-1].upper()}")

    verify_weights_unchanged(model, snapshot)

if __name__ == "__main__":
    main()
