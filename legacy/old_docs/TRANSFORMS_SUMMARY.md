# Transform Management - Summary

## What I've Implemented

### 1. **Config-Based Transform Management**
   - Added transform parameters to `ExperimentConfig`
   - Image size, normalization settings, augmentation options
   - All saved with experiment for reproducibility

### 2. **Transform Builder Utility** (`src/li_4yp/utils/transforms.py`)
   - `build_transform()`: Build from config parameters
   - `get_imagenet_transform()`: Quick ImageNet preset
   - `get_basic_transform()`: Simple resize + ToTensor
   - `get_transform_from_preset()`: Use named presets

### 3. **Updated Training Script** (`scripts/train.py`)
   - Automatically builds transforms from config
   - No manual transform management needed

## How to Use

### Option 1: Configuration File (Recommended)

**1. Create config with transforms:**

```yaml
# config/experiments/my_experiment.yaml
image_size: [224, 224]
normalize: true
normalize_mean: [0.485, 0.456, 0.406]
normalize_std: [0.229, 0.224, 0.225]
augmentation:
  random_horizontal_flip: true
  flip_prob: 0.5
  color_jitter: true
  brightness: 0.2
```

**2. Run experiment:**

```bash
python scripts/train.py --config_path config/experiments/my_experiment.yaml
```

That's it! Transforms are built automatically.

### Option 2: Programmatic

```python
from li_4yp.experiments import ExperimentConfig, Experiment
from li_4yp.utils import build_transform
from li_4yp.data import DigitalSwatchDataset

# Config with transform settings
config = ExperimentConfig(
    name="my_exp",
    model_type="pytorch",
    model_name="mobilenet_v3",
    image_size=(224, 224),
    normalize=True,
    normalize_mean=(0.485, 0.456, 0.406),
    normalize_std=(0.229, 0.224, 0.225),
    augmentation={
        "random_horizontal_flip": True,
        "color_jitter": True
    }
)

# Build transform
transform = build_transform(
    image_size=config.image_size,
    normalize=config.normalize,
    normalize_mean=config.normalize_mean,
    normalize_std=config.normalize_std,
    augmentation=config.augmentation
)

# Run experiment
exp = Experiment(config)
exp.run(DigitalSwatchDataset, transform=transform)
```

### Option 3: Quick Presets

```python
from li_4yp.utils import get_transform_from_preset

# Use a preset
transform = get_transform_from_preset("imagenet", image_size=(224, 224))

# Available: "imagenet", "basic", "mobilenet", "none"
```

## Key Features

### ✅ Reproducibility
All transform settings saved in experiment config:
```json
{
  "image_size": [224, 224],
  "normalize": true,
  "normalize_mean": [0.485, 0.456, 0.406],
  "augmentation": {...}
}
```

### ✅ Flexibility
Control everything through config:
- Image size
- Normalization (on/off, custom mean/std)
- Augmentation (flip, rotation, color jitter, etc.)

### ✅ Consistency
Same transform used across training, validation, test:
```python
# Training (with augmentation)
train_transform = build_transform(..., is_training=True)

# Validation (no augmentation)  
val_transform = build_transform(..., is_training=False)
```

### ✅ Extensibility
Easy to add new augmentations in `transforms.py`

## Example Configs

### Minimal
```yaml
image_size: [224, 224]
normalize: true
normalize_mean: [0.485, 0.456, 0.406]
normalize_std: [0.229, 0.224, 0.225]
augmentation: {}
```

### With Augmentation
```yaml
image_size: [224, 224]
normalize: true
normalize_mean: [0.485, 0.456, 0.406]
normalize_std: [0.229, 0.224, 0.225]
augmentation:
  random_horizontal_flip: true
  flip_prob: 0.5
  color_jitter: true
  brightness: 0.2
  contrast: 0.2
```

## Available Augmentations

- `random_horizontal_flip`: Randomly flip images
- `random_rotation`: Rotate by random degrees
- `random_crop`: Random crop then resize
- `color_jitter`: Adjust brightness/contrast/saturation/hue

## Best Practices

1. **Use ImageNet normalization for pretrained models**
   ```yaml
   normalize_mean: [0.485, 0.456, 0.406]
   normalize_std: [0.229, 0.224, 0.225]
   ```

2. **Document transforms in config** for reproducibility

3. **Disable augmentation for validation/test**
   ```python
   build_transform(..., is_training=False)
   ```

4. **Match image size to model requirements**
   - MobileNet/ResNet: 224×224
   - EfficientNet-B0: 224×224
   - EfficientNet-B7: 600×600

## Complete Example

See `config/experiments/mobilenet_augmented.yaml` for a full example with:
- ImageNet normalization
- Data augmentation
- All experiment settings

Run it:
```bash
python scripts/train.py --config_path config/experiments/mobilenet_augmented.yaml
```

## Documentation

- Full guide: `TRANSFORMS_GUIDE.md`
- Example configs: `config/experiments/`
- Code: `src/li_4yp/utils/transforms.py`

This system ensures your transforms are always documented, reproducible, and easy to modify! 🎯
