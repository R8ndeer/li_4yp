# Presentation Repo

Minimal presentation slice for the Marshall Wace technical interview.

## Entry Point

```bash
python scripts/train.py --config_path config/experiments/attentive_statnet.yaml
```

## Pipeline

`config -> dataset -> transforms -> model -> training -> evaluation -> logging`

Key files:
- `scripts/train.py`
- `config/experiments/attentive_statnet.yaml`
- `src/li_4yp/experiments/experiment.py`
- `src/li_4yp/data/datasets.py`
- `src/li_4yp/utils/transforms.py`
- `src/li_4yp/models/attentionstatnet.py`
- `src/li_4yp/training/metrics.py`

## Layout

- `src/li_4yp/`: runnable training pipeline
- `data/masterlist_v7/`: runtime dataset
- `outputs/attentive_statnet_example/`: example saved run
- `legacy/`: non-presentation material moved out of the main path
