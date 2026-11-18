
import os
import sys
import subprocess
from pathlib import Path

# Experiment project directory
ROOT = Path(__file__).resolve().parent

# Parent directory containing mbt_specialized.py
MODEL_ROOT = ROOT.parent

# Add both directories to Python's import path
env = os.environ.copy()

paths = [str(ROOT), str(MODEL_ROOT)]
existing_path = env.get("PYTHONPATH", "")

if existing_path:
    paths.append(existing_path)

env["PYTHONPATH"] = os.pathsep.join(paths)

experiments = [
    "experiments/E0_baseline.py",
    "experiments/E1_sensor_permutation.py",
    "experiments/E2_feature_permutation.py",
    "experiments/E3_sensor_ablation.py",
    "experiments/E4_feature_order_shuffle.py",
    "experiments/E5_all_feature_shuffle.py",
]

for experiment in experiments:
    script = ROOT / experiment

    print(f"\n========== {experiment} ==========")

    subprocess.run(
        [sys.executable, str(script)],
        cwd=str(ROOT),
        env=env,
        check=True
    )

print("\nAll experiments completed.")
