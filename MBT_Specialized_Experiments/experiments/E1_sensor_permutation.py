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

    # Infer each contiguous sensor block from the feature-name prefixes.
    prefixes = ["radar_", "df_", "tv_", "it_", "pan_"]
    groups = {}
    for prefix in prefixes:
        indices = [i for i, name in enumerate(FEATURE_NAMES) if name.startswith(prefix)]
        if not indices or indices != list(range(min(indices), max(indices) + 1)):
            raise ValueError(f"Features for {prefix} are missing or not contiguous.")
        groups[prefix[:-1].upper()] = slice(min(indices), max(indices) + 1)

    for sensor, sl in groups.items():
        changed = X.copy()
        row_order = rng.permutation(len(X))
        # Permute complete sensor vectors across samples, preserving within-sensor relations.
        changed[:, sl] = X[row_order, sl]
        evaluate(model, changed, y, f"E1_permute_{sensor}")

    verify_weights_unchanged(model, snapshot)

if __name__ == "__main__":
    main()
