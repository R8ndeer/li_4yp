import argparse
from pathlib import Path

from li_4yp.data import DigitalSwatchDataset, HybridSwatchDataset
from li_4yp.experiments import Experiment, ExperimentConfig, ModelRegistry


def main():
    """Main training script."""
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to experiment config JSON/YAML file",
    )
    args = ap.parse_args()

    # load experiment config and validate
    config_path = Path(args.config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    if config_path.suffix in {".yaml", ".yml"}:
        config = ExperimentConfig.from_yaml(config_path)
    elif config_path.suffix == ".json":
        config = ExperimentConfig.from_json(config_path)
    else:
        raise ValueError(
            "Unsupported config file format: config file must be JSON or YAML format."
        )

    # verify model
    if not ModelRegistry.is_registered(config.model_name):
        raise ValueError(
            f"Model '{config.model_name}' is not registered in ModelRegistry."
        )

    # check paths
    paths = {
        "data_dir": Path(config.data_dir),
        "csv_file": Path(config.data_dir) / config.csv_file,
        "save_dir": Path(config.save_dir),
    }

    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Required path '{name}' does not exist: {path}")

    # create and run experiment
    experiment = Experiment(config)
    if config.dataset_class.lower() == "HybridSwatchDataset".lower():
        experiment.run(dataset_class=HybridSwatchDataset, **config.dataset_params)
    else:
        experiment.run(dataset_class=DigitalSwatchDataset, **config.dataset_params)


if __name__ == "__main__":
    main()
