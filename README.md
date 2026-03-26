# li_4yp Presentation Slice

This repository is a cleaned presentation slice of the original 4YP hair-shade prediction project. The active code path is intentionally narrow so it is easy to run, explain, and discuss in a technical interview.

## Active Entry Point

Run the main experiment from the repository root:

```bash
python scripts/train.py --config_path config/experiments/attentive_statnet.yaml
```

Run the sklearn example path from the repository root:

```bash
python scripts/train.py --config_path config/experiments/example_sklearn.yaml
```

The primary presentation path is:

`config -> experiment -> dataset -> transforms -> model -> loss -> evaluation -> logging`

## Main Files To Present

- `config/experiments/attentive_statnet.yaml`
- `scripts/train.py`
- `src/li_4yp/experiments/config.py`
- `src/li_4yp/experiments/experiment.py`
- `src/li_4yp/experiments/model_registry.py`
- `src/li_4yp/experiments/loss_registry.py`
- `src/li_4yp/data/datasets.py`
- `src/li_4yp/utils/transforms.py`
- `src/li_4yp/utils/transform_presets.py`
- `src/li_4yp/models/attentionstatnet.py`
- `src/li_4yp/training/metrics.py`

## Repository Layout

- `src/li_4yp/`: active presentation package
- `config/experiments/`: runnable and validation-demo configs
- `scripts/`: command-line entry points for the presentation slice
- `tests/`: config and branch-coverage tests for the active pipeline
- `legacy/`: archived material kept for reference, not part of the active presentation flow

## Environment

The project uses a `src/` layout and is intended to be installed in editable mode:

```bash
conda env create -f environment.yml
conda activate li_4yp
pip install -e .
```

If the environment already exists, update it with:

```bash
conda env update -f environment.yml --prune
conda activate li_4yp
```

You can also install directly into an existing Python environment with:

```bash
pip install -e .
```


## Notes

- The active PyTorch presentation run uses `attentive_pixel_stat_net`.
- The sklearn path is supported in the presentation slice via `config/experiments/example_sklearn.yaml`.
- Files under `legacy/` are intentionally out of the main execution path.
