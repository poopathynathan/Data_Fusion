import numpy as np
from common.config import RANDOM_SEED
from common.data import load_test_data
from common.model import load_trained_model, verify_weights_unchanged
from common.evaluate import evaluate

def main():
    X, y = load_test_data()
    model, snapshot = load_trained_model()
    rng = np.random.default_rng(RANDOM_SEED)

    # Deliberately mismatches values with the model's learned feature-position
    # embeddings and fixed domain routing. This is a stress test only.
    permutation = rng.permutation(X.shape[1])
    changed = X[:, permutation]
    evaluate(model, changed, y, "E5_global_feature_shuffle")

    verify_weights_unchanged(model, snapshot)

if __name__ == "__main__":
    main()
