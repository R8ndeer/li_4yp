# Scripts

This directory contains the active command-line entry points for the presentation slice.

## Available Script

### `train.py`

Runs a config-driven experiment from the active presentation pipeline.

Example:

```bash
python scripts/train.py --config_path config/experiments/attentive_statnet.yaml
```

What it does:

- loads and validates the experiment config
- registers models for the selected model family
- resolves the dataset class for PyTorch runs
- creates and runs `Experiment`

## Common Configs

- Main PyTorch presentation run:

```bash
python scripts/train.py --config_path config/experiments/attentive_statnet.yaml
```

- Example sklearn run:

```bash
python scripts/train.py --config_path config/experiments/example_sklearn.yaml
```

- Validation-demo configs:

```bash
python scripts/train.py --config_path config/experiments/failure_missing_train_split.yaml
python scripts/train.py --config_path config/experiments/failure_missing_dataset_class.yaml
```

These failure configs are intentionally invalid and are useful for checking user-facing validation errors.

## Requirements

Run scripts from the repository root. The package should be available either through:

```bash
pip install -e .
```

or:

```bash
PYTHONPATH=src python scripts/train.py --config_path ...
```
