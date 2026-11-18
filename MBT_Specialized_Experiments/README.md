# Specialized MBT — Single Checkpoint Experiments

This corrected project uses the specialized architecture from `mbt_specialized.py`,
not `mbt_sensor_classifier.py`.

## Required original files in this folder

- `mbt_specialized.py`
- `sensor_data_calibrated_3classes_5000.csv`
- `mbt_specialized_3layers.pt`
- `mbt_specialized_3layers_scaler.joblib`

The checkpoint and scaler names match the supplied `train_mbt_specialized.py`
script when trained with `--layers 3`.

## Model configuration

`SpecializedBottleneckTransformer`:
- 54 scalar feature tokens
- 4 domain-guided bottleneck tokens
- d_model=128
- 4 attention heads
- 3 fusion layers
- FFN dimension=512
- dropout=0.1
- 3 output classes

## Run

Install:
`python -m pip install numpy pandas scikit-learn torch joblib`

Run an individual experiment from this directory:
- `python -m experiments.E0_baseline`
- `python -m experiments.E1_sensor_permutation`
- `python -m experiments.E2_feature_permutation`
- `python -m experiments.E3_sensor_ablation`
- `python -m experiments.E4_feature_order_shuffle`
- `python -m experiments.E5_all_feature_shuffle`

Or run all:
`python run_all.py`

All scripts load the same trained checkpoint and scaler. They do not train or
save model weights. The model file and training script are not included; copy
your original files into this project directory.

## Critical validation

E0 must reproduce the held-out result from the specialized training run.
The training script uses the CSV columns in `FEATURE_NAMES` order and a
70/15/15 stratified split with seed 42. The scaler is fitted only on training
rows. Do not substitute the original MBT scaler or the NPZ-based baseline
preprocessing unless you verify exact feature identity and order.

E1/E2 permute values across samples and test dependence on feature information.
E3 replaces a sensor block with standardized zero (approximately its training
mean). E4/E5 change the feature-to-position mapping at inference; they are
deliberate stress tests, not alternative trained models.
"# Fusion" 
