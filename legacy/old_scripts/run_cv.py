import os
import sys
import yaml
import subprocess
import argparse
from pathlib import Path

# Define the experiments you want to run
CONFIGS_TO_RUN = [
    # "config/experiments/REPORT_resnet_2.yaml",
    # "config/experiments/REPORT_attentivestatnet.yaml",
    # "config/experiments/REPORT_resnet.yaml",
    # "config/experiments/REPORT_attentivestatnetonemoment.yaml",
    "config/experiments/REPORT_statnet.yaml",
]

# The folds defined in your golden_dataset_5fold.csv
FOLDS = [0, 1, 2, 3, 4]


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_yaml(data, path):
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)


def run_fold(base_config_path, fold, dry_run=False):
    config_path = Path(base_config_path)
    if not config_path.exists():
        print(f"Error: Config file {config_path} not found.")
        return

    # 1. Load original config
    config_data = load_yaml(config_path)
    model_name = config_data.get("name", config_path.stem)

    # 2. Setup Fold-Specific Paths
    # We construct the save path here: experiments/ModelName/fold_X
    base_save_dir = Path("experiments") / model_name / f"fold_{fold}"

    # --- FIX 1: Create the directory immediately ---
    # This prevents train.py from crashing due to FileNotFoundError
    if not dry_run:
        os.makedirs(base_save_dir, exist_ok=True)

    # 3. Inject Dynamic Configuration
    config_data["val_fold_idx"] = fold
    config_data["name"] = f"{model_name}_fold{fold}"
    config_data["save_dir"] = str(base_save_dir)
    config_data["csv_file"] = (
        "golden_dataset_5fold.csv"  # Ensure this file is in your data_dir!
    )

    # --- FIX 2: Save Config Permanently in the Experiment Folder ---
    # Instead of a temp file in root, save it where it belongs.
    fold_config_path = base_save_dir / "config.yaml"

    if not dry_run:
        save_yaml(config_data, fold_config_path)

    print(f"\n" + "=" * 60)
    print(f"🚀 RUNNING: {model_name} | FOLD {fold}/4 ({fold + 1}/5)")
    print(f"📂 Output: {base_save_dir}")
    print(f"📄 Config: {fold_config_path}")
    print("=" * 60 + "\n")

    # Point train.py to the config we just saved inside the folder
    cmd = [sys.executable, "scripts/train.py", "--config_path", str(fold_config_path)]

    if dry_run:
        print(f"Dry run command: {' '.join(cmd)}")
    else:
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ Error occurred during training fold {fold} for {model_name}.")
            print(e)
            # Optional: If training fails immediately, you might want to keep the config
            # to debug it, so we don't delete it here either.


def main():
    parser = argparse.ArgumentParser(description="Run 5-Fold Cross Validation")
    parser.add_argument(
        "--dry-run", action="store_true", help="Print commands without running them"
    )
    args = parser.parse_args()

    # Ensure root experiments dir exists
    if not args.dry_run:
        os.makedirs("experiments", exist_ok=True)

    for config_file in CONFIGS_TO_RUN:
        for fold in FOLDS:
            run_fold(config_file, fold, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
