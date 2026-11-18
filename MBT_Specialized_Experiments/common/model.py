
import torch

from mbt_specialized import SpecializedBottleneckTransformer
from .config import MODEL_PATH, DEVICE


def load_trained_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model checkpoint not found: {MODEL_PATH}"
        )

    print(f"\nLoading trained model: {MODEL_PATH}")

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    # Reconstruct the architecture from the saved checkpoint
    model = SpecializedBottleneckTransformer(
        num_classes=checkpoint["num_classes"],
        d_model=checkpoint["d_model"],
        n_heads=checkpoint["n_heads"],
        num_layers=checkpoint["num_layers"],
        ff_dim=checkpoint["ff_dim"],
        dropout=checkpoint["dropout"]
    )

    # Load the original trained weights
    model.load_state_dict(checkpoint["model_state_dict"])

    model.to(DEVICE)
    model.eval()

    print("Model loaded successfully.")
    print(f"Architecture layers: {checkpoint['num_layers']}")
    print(f"Device: {DEVICE}")

    return model, checkpoint


def verify_weights_unchanged(model, snapshot):

    if "model_state_dict" not in snapshot:
        raise KeyError(
            "Snapshot does not contain model_state_dict."
        )

    original_weights = snapshot["model_state_dict"]
    current_weights = model.state_dict()

    # Check parameter names
    if original_weights.keys() != current_weights.keys():
        raise RuntimeError(
            "Model architecture or parameter names have changed."
        )

    changed_parameters = []

    # Compare every parameter with the original checkpoint
    for name, original_tensor in original_weights.items():

        current_tensor = current_weights[name]

        original_tensor = original_tensor.detach().cpu()
        current_tensor = current_tensor.detach().cpu()

        if not torch.equal(original_tensor, current_tensor):
            changed_parameters.append(name)

    if changed_parameters:
        raise RuntimeError(
            "Trained model weights have changed!\n"
            f"Changed parameters: {changed_parameters}"
        )

    print("WEIGHT VERIFICATION PASSED")
    print("All model weights match the original trained checkpoint.")

    return True
