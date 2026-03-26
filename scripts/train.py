"""Entry point for running config-driven training experiments."""

import argparse
from pathlib import Path

from li_4yp.experiments import ExperimentConfig, ModelRegistry
from li_4yp.experiments.model_registry import (
    register_pytorch_models,
    register_sklearn_models,
)
from li_4yp.experiments.loss_registry import LossRegistry, register_default_losses


def _load_config(config_path: Path) -> ExperimentConfig:
    """Load an experiment config from a supported file type."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    if config_path.suffix in {".yaml", ".yml"}:
        return ExperimentConfig.from_yaml(config_path)
    if config_path.suffix == ".json":
        return ExperimentConfig.from_json(config_path)

    raise ValueError(
        f"Unsupported config file format '{config_path.suffix}'. "
        "Expected one of: .yaml, .yml, .json."
    )


def _validate_runtime_paths(config: ExperimentConfig) -> None:
    """Ensure the config references files and directories that exist."""
    save_dir = Path(config.save_dir)
    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)

    required_paths = {
        "data_dir": Path(config.data_dir),
        "csv_file": Path(config.data_dir) / config.csv_file,
    }

    for name, path in required_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Required path '{name}' does not exist: {path}")


def main():
    """Main training script."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to experiment config JSON/YAML file",
    )
    args = parser.parse_args()

    config_path = Path(args.config_path)
    config = _load_config(config_path)

    if config.model_type == "pytorch":
        register_pytorch_models()
        register_default_losses()
    elif config.model_type == "sklearn":
        register_sklearn_models()

    if not ModelRegistry.is_registered(config.model_name):
        raise ValueError(
            f"Model '{config.model_name}' is not registered in ModelRegistry. "
            f"Available models: {list(ModelRegistry.list_models().keys())}"
        )
    if not LossRegistry.is_registered(config.loss_name):
        raise ValueError(
            f"Loss '{config.loss_name}' is not registered in LossRegistry. "
            f"Available losses: {list(LossRegistry.list_losses().keys())}"
        )

    _validate_runtime_paths(config)

    from li_4yp.experiments import Experiment

    experiment = Experiment(config)
    experiment.run()


if __name__ == "__main__":
    main()
