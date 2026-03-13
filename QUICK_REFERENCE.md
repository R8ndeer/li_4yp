# Experiment System - Quick Reference

## One-Line Experiments

```python
# PyTorch
Experiment(ExperimentConfig.from_yaml("config.yaml")).run(DigitalSwatchDataset, transform=tsfm)

# Scikit-learn
Experiment(ExperimentConfig.from_yaml("config.yaml")).run(DigitalSwatchDataset)
```

## Common Tasks

### Create a New Experiment

```python
from li_4yp.experiments import Experiment, ExperimentConfig
from li_4yp.data import DigitalSwatchDataset

config = ExperimentConfig(
    name="my_exp",
    model_type="pytorch",  # or "sklearn"
    model_name="cnn_rnn",  # from registry
    data_dir="data/masterlist_v6",
    csv_file="labels.csv",
    batch_size=32,
    num_epochs=100,
    learning_rate=0.001
)

exp = Experiment(config)
metrics = exp.run(DigitalSwatchDataset)
```

### List Available Models

```python
from li_4yp.experiments import ModelRegistry
print(ModelRegistry.list_models())
```

### Register a Custom Model

```python
ModelRegistry.register(
    name="my_model",
    constructor=lambda **kwargs: MyModel(**kwargs),
    description="Description"
)
```

### Load Config from File

```python
config = ExperimentConfig.from_yaml("path/to/config.yaml")
# or
config = ExperimentConfig.from_json("path/to/config.json")
```

### Save Config to File

```python
config.to_yaml("path/to/config.yaml")
config.to_json("path/to/config.json")
```

### Run Batch Experiments

```python
for lr in [1e-4, 1e-3, 1e-2]:
    config = ExperimentConfig(
        name=f"lr_{lr}",
        learning_rate=lr,
        # ... other params
    )
    exp = Experiment(config)
    exp.run(DigitalSwatchDataset)
```

### Load Saved Model

```python
import torch
model = ModelRegistry.get("my_model", **params)
model.load_state_dict(torch.load("experiments/.../model_best.pth"))

# or sklearn
import pickle
with open("experiments/.../model.pkl", 'rb') as f:
    model = pickle.load(f)
```

### Compare Experiments

```python
import pandas as pd
from pathlib import Path
import json

results = []
for exp in Path("experiments").glob("*/summary.json"):
    with open(exp) as f:
        results.append(json.load(f))

df = pd.DataFrame(results)
print(df.sort_values("final_metrics.base_acc", ascending=False))
```

## Configuration Template

```yaml
name: "experiment_name"
description: "What this tests"
tags: ["tag1", "tag2"]

model_type: "pytorch"  # or "sklearn"
model_name: "model_from_registry"
model_params:
  param1: value1

data_dir: "data/dataset"
csv_file: "labels.csv"
train_split: 0.8
random_seed: 42

# PyTorch only
batch_size: 32
num_epochs: 100
learning_rate: 0.001
optimizer: "Adam"
early_stopping_patience: 10

# Evaluation
evaluator_params:
  eos_token: -1
  weights: [0.3, 0.1]
  tol_base: 4

save_dir: "experiments"
save_model: true
save_predictions: true
```

## Pre-registered Models

### PyTorch
- `cnn_rnn`: CNN-RNN sequential model
- `multi_output_cnn`: Multi-output CNN

### Scikit-learn
- `random_forest`: Random Forest
- `extra_trees`: Extra Trees
- `mlp`: Multi-layer Perceptron

## Output Files

Every experiment creates:
- `config.json` - Configuration
- `experiment.log` - Logs
- `history.csv` - Training curves
- `metrics.csv` - All metrics
- `summary.json` - Final results
- `model_best.pth` or `model.pkl` - Trained model
- `val_predictions.csv` - Predictions

## Tips

1. Always use meaningful experiment names
2. Tag experiments for easy filtering
3. Save configs to version control
4. Use `random_seed` for reproducibility
5. Check logs if something fails
6. Compare experiments using `summary.json`

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Model not found | Register model first or check spelling |
| CUDA OOM | Reduce batch_size or use device="cpu" |
| Dataset not found | Check data_dir and csv_file paths |
| Import error | Make sure li_4yp is installed |

## Examples

See:
- `notebooks/21_experiment_system_demo.ipynb` - Complete demo
- `src/li_4yp/experiments/README.md` - Full documentation
- `config/experiments/` - Example configs
