from common.data import load_test_data
from common.model import load_trained_model, verify_weights_unchanged
from common.evaluate import evaluate

def main():
    X, y = load_test_data()
    model, snapshot = load_trained_model()
    evaluate(model, X, y, "E0_baseline_specialized")
    verify_weights_unchanged(model, snapshot)

if __name__ == "__main__":
    main()
