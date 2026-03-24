# Experiment Management System

A comprehensive experiment tracking and management system for reproducible machine learning experiments.

## Features

- ✅ **Unified Interface**: Works with both PyTorch and scikit-learn models
- ✅ **Automatic Logging**: Tracks metrics, losses, configurations, and predictions
- ✅ **Model Registry**: Easy model registration and instantiation
- ✅ **Reproducibility**: Automatic seed setting and configuration saving
- ✅ **Extensibility**: Easy to add new models and metrics
- ✅ **Experiment Comparison**: Structured output for easy comparison

## Quick Start

### 1. Basic PyTorch Experiment

```python
from li_4yp.experiments import Experiment, ExperimentConfig
from li_4yp.data import DigitalSwatchDataset
from torchvision import transforms

# Load configuration
config = ExperimentConfig.from_yaml("config/experiments/example_pytorch.yaml")

# Create experiment
experiment = Experiment(config)

# Setup transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Run experiment
final_metrics = experiment.run(
    dataset_class=DigitalSwatchDataset,
    transform=transform
)

print("Final Metrics:", final_metrics)
```

### 2. Basic Scikit-learn Experiment

```python
from li_4yp.experiments import Experiment, ExperimentConfig
from li_4yp.data import DigitalSwatchDataset

# Load configuration
config = ExperimentConfig.from_yaml("config/experiments/example_sklearn.yaml")

# Create experiment
experiment = Experiment(config)

# Run experiment
final_metrics = experiment.run(
    dataset_class=DigitalSwatchDataset
)

print("Final Metrics:", final_metrics)
```

### 3. Programmatic Configuration

```python
from li_4yp.experiments import Experiment, ExperimentConfig

# Create config programmatically
config = ExperimentConfig(
    name="my_experiment",
    description="Testing a new model",
    tags=["test", "custom"],
    model_type="pytorch",
    model_name="multi_output_cnn",
    model_params={"vocab_size": 12},
    dataset_name="masterlist_v6",
    data_dir="data/masterlist_v6",
    csv_file="masterlist_v6_full_features.csv",
    batch_size=32,
    num_epochs=50,
    learning_rate=0.001
)

# Run experiment
experiment = Experiment(config)
metrics = experiment.run(dataset_class=DigitalSwatchDataset, transform=transform)
```

## Model Registry

### Built-in Models

The system comes with pre-registered models:

**PyTorch Models:**
- `cnn_rnn`: CNN-RNN model for sequential prediction
- `multi_output_cnn`: Multi-output CNN for parallel prediction

**Scikit-learn Models:**
- `random_forest`: Random Forest classifier
- `extra_trees`: Extra Trees classifier
- `mlp`: Multi-layer Perceptron

### Registering Custom Models

#### PyTorch Models

```python
from li_4yp.experiments import ModelRegistry
import torch.nn as nn

class MyCustomModel(nn.Module):
    def __init__(self, hidden_dim=256, num_classes=12):
        super().__init__()
        self.fc = nn.Linear(512, hidden_dim)
        self.out = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        x = torch.relu(self.fc(x))
        return self.out(x)

# Register the model
ModelRegistry.register(
    name="my_custom_model",
    constructor=lambda **kwargs: MyCustomModel(
        hidden_dim=kwargs.get('hidden_dim', 256),
        num_classes=kwargs.get('num_classes', 12)
    ),
    description="My custom PyTorch model"
)

# Use in experiment
config = ExperimentConfig(
    name="custom_model_exp",
    model_type="pytorch",
    model_name="my_custom_model",
    model_params={"hidden_dim": 512, "num_classes": 12},
    # ... other config
)
```

#### Scikit-learn Models

```python
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.multioutput import MultiOutputClassifier

ModelRegistry.register(
    name="gradient_boosting",
    constructor=lambda **kwargs: MultiOutputClassifier(
        GradientBoostingClassifier(
            n_estimators=kwargs.get('n_estimators', 100),
            learning_rate=kwargs.get('learning_rate', 0.1),
            random_state=kwargs.get('random_state', 42)
        )
    ),
    description="Gradient Boosting classifier"
)
```

## Configuration File Format

Configuration files use YAML format. Here's a complete example:

```yaml
# Experiment metadata
name: "my_experiment"
description: "Detailed description of what this experiment tests"
tags:
  - "tag1"
  - "tag2"

# Model configuration
model_type: "pytorch"  # or "sklearn"
model_name: "model_name_from_registry"
model_params:
  param1: value1
  param2: value2

# Data configuration
dataset_name: "dataset_name"
data_dir: "path/to/data"
csv_file: "labels.csv"
train_split: 0.8
random_seed: 42

# Training configuration (PyTorch only)
batch_size: 32
num_epochs: 100
learning_rate: 0.001
optimizer: "Adam"
optimizer_params:
  weight_decay: 0.0001
scheduler: "ReduceLROnPlateau"  # optional
scheduler_params:
  mode: "min"
  factor: 0.5
  patience: 5
early_stopping_patience: 10

# Loss configuration
loss_function: "CrossEntropyLoss"
loss_weights: [1.0, 1.0, 1.0]

# Evaluation configuration
evaluator_params:
  eos_token: -1
  weights: [0.3, 0.1]
  tol_base: 4

# Output configuration
save_dir: "experiments"
save_model: true
save_predictions: true
save_history: true

# Hardware configuration
device: "auto"  # "auto", "cpu", or "cuda"
num_workers: 0
```

## Output Structure

Each experiment creates a timestamped directory:

```
experiments/
└── 20251110_143022_my_experiment/
    ├── config.json              # Experiment configuration
    ├── experiment.log           # Detailed logs
    ├── metrics.csv              # All metrics logged
    ├── history.csv              # Training history (epoch-by-epoch)
    ├── model_info.json          # Model architecture info
    ├── data_info.json           # Dataset information
    ├── summary.json             # Final summary
    ├── model_name_best.pth      # Best model checkpoint (PyTorch)
    ├── model_name.pkl           # Trained model (sklearn)
    ├── val_predictions.csv      # Validation predictions
    └── test_predictions.csv     # Test predictions (if applicable)
```

## Advanced Usage

### Custom Evaluator

```python
from li_4yp.training import ShadeEvaluator

# Modify evaluator parameters
config.evaluator_params = {
    "eos_token": -1,
    "weights": (0.4, 0.2, 0.1),  # Custom weights for hierarchical score
    "tol_base": 5  # Different tolerance threshold
}
```

### Custom Loss Function

For PyTorch models with custom loss functions:

```python
import torch.nn as nn

class CustomLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
    
    def forward(self, outputs, targets):
        # Custom loss logic
        return self.ce(outputs, targets)

# Modify the Experiment class to use custom loss
# (or extend the Experiment class)
```

### Batch Experiments

Run multiple experiments with different configurations:

```python
from pathlib import Path

config_dir = Path("config/experiments")
results = {}

for config_file in config_dir.glob("*.yaml"):
    config = ExperimentConfig.from_yaml(config_file)
    experiment = Experiment(config)
    metrics = experiment.run(dataset_class=DigitalSwatchDataset, transform=transform)
    results[config.name] = metrics

# Compare results
import pandas as pd
df = pd.DataFrame(results).T
print(df)
df.to_csv("experiments/comparison.csv")
```

### Loading Saved Models

```python
import torch

# PyTorch model
model = ModelRegistry.get("mobilenet_v3", num_classes=[12, 11, 11])
model.load_state_dict(torch.load("experiments/20251110_143022_my_experiment/mobilenet_v3_best.pth"))
model.eval()

# Scikit-learn model
import pickle
with open("experiments/20251110_143022_my_experiment/random_forest.pkl", 'rb') as f:
    model = pickle.load(f)
```

## Extending the System

### Adding Custom Metrics

Extend the `ShadeEvaluator` class:

```python
from li_4yp.training import ShadeEvaluator

class CustomEvaluator(ShadeEvaluator):
    def _custom_metric(self, preds, labels):
        # Your custom metric logic
        return score
    
    def summary(self):
        summary = super().summary()
        # Add custom metrics
        summary["custom_metric"] = np.mean(self.custom_scores)
        return summary
```

### Adding New Model Types

You can extend the system to support other frameworks (e.g., JAX, TensorFlow):

```python
# In experiments/experiment.py, add methods like:
def train_tensorflow(self):
    # TensorFlow training logic
    pass

def evaluate_tensorflow(self, data_loader):
    # TensorFlow evaluation logic
    pass
```

## Best Practices

1. **Always use configuration files** for reproducibility
2. **Tag experiments** with meaningful labels
3. **Document** your experiments with clear descriptions
4. **Version control** your config files
5. **Compare experiments** systematically using the structured output
6. **Save predictions** for post-hoc analysis
7. **Use consistent random seeds** across experiments
8. **Archive important experiments** with their configs and results

## Troubleshooting

### Issue: CUDA out of memory
- Reduce `batch_size` in config
- Reduce model size in `model_params`
- Set `device: "cpu"` for testing

### Issue: Model not registered
- Make sure to import and register custom models before creating experiment
- Check model name spelling in config

### Issue: Dataset not found
- Verify `data_dir` and `csv_file` paths in config
- Use absolute paths if needed

## Examples

See the `notebooks/` directory for complete examples:
- `example_pytorch_experiment.ipynb` - PyTorch model training
- `example_sklearn_experiment.ipynb` - Scikit-learn model training
- `example_batch_experiments.ipynb` - Running multiple experiments
- `example_custom_models.ipynb` - Registering and using custom models
