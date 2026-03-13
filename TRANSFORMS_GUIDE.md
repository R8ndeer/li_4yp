# Transform Management Guide

## Overview

Transforms are managed through the configuration system and utility functions, providing flexibility and reproducibility.

## Approach 1: Configuration-Based (Recommended)

Define transforms in your experiment config file:

```yaml
# config/experiments/my_experiment.yaml
image_size: [224, 224]
normalize: true
normalize_mean: [0.485, 0.456, 0.406]
normalize_std: [0.229, 0.224, 0.225]
augmentation:
  random_horizontal_flip: true
  flip_prob: 0.5
  random_rotation: true
  rotation_degrees: 10
  color_jitter: true
  brightness: 0.2
  contrast: 0.2
  saturation: 0.2
  hue: 0.1
```

Then in your script:

```python
from li_4yp.utils import build_transform
from li_4yp.experiments import ExperimentConfig

config = ExperimentConfig.from_yaml("config/experiments/my_experiment.yaml")

# Training transform (with augmentation)
train_transform = build_transform(
    image_size=config.image_size,
    normalize=config.normalize,
    normalize_mean=config.normalize_mean,
    normalize_std=config.normalize_std,
    augmentation=config.augmentation,
    is_training=True  # Enables augmentation
)

# Validation transform (no augmentation)
val_transform = build_transform(
    image_size=config.image_size,
    normalize=config.normalize,
    normalize_mean=config.normalize_mean,
    normalize_std=config.normalize_std,
    augmentation=config.augmentation,
    is_training=False  # Disables augmentation
)
```

## Approach 2: Presets

Use predefined transform presets:

```python
from li_4yp.utils import get_transform_from_preset

# Available presets: "imagenet", "basic", "mobilenet", "none"
transform = get_transform_from_preset("imagenet", image_size=(224, 224))
```

### Available Presets

1. **`imagenet`**: Standard ImageNet normalization
   ```python
   transforms.Compose([
       transforms.Resize((224, 224)),
       transforms.ToTensor(),
       transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
   ])
   ```

2. **`basic`**: Just resize and convert to tensor
   ```python
   transforms.Compose([
       transforms.Resize((224, 224)),
       transforms.ToTensor()
   ])
   ```

3. **`mobilenet`**: Same as ImageNet (compatible with pretrained MobileNet)

4. **`none`**: Same as basic (no normalization)

## Approach 3: Direct Construction

For full control, construct transforms directly:

```python
from torchvision import transforms

custom_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
```

## Integration with Experiment System

### In train.py Script

The training script automatically builds transforms from config:

```bash
python scripts/train.py --config_path config/experiments/my_experiment.yaml
```

The script will:
1. Load config
2. Build transform from config parameters
3. Pass it to the experiment

### In Jupyter Notebooks

```python
from li_4yp.experiments import Experiment, ExperimentConfig
from li_4yp.utils import build_transform
from li_4yp.data import DigitalSwatchDataset

# Load config
config = ExperimentConfig.from_yaml("config/experiments/my_experiment.yaml")

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

### Programmatic Configuration

```python
config = ExperimentConfig(
    name="my_experiment",
    model_type="pytorch",
    model_name="mobilenet_v3",
    # Transform settings
    image_size=(224, 224),
    normalize=True,
    normalize_mean=(0.485, 0.456, 0.406),
    normalize_std=(0.229, 0.224, 0.225),
    augmentation={
        "random_horizontal_flip": True,
        "flip_prob": 0.5,
        "color_jitter": True,
        "brightness": 0.2
    }
)
```

## Available Augmentation Options

### Geometric Transforms

```yaml
augmentation:
  random_horizontal_flip: true
  flip_prob: 0.5  # Probability of flipping
  
  random_rotation: true
  rotation_degrees: 15  # Rotation range in degrees
  
  random_crop: true
  crop_size: [224, 224]  # Crop size (will be resized back)
```

### Color Transforms

```yaml
augmentation:
  color_jitter: true
  brightness: 0.2  # Brightness factor
  contrast: 0.2    # Contrast factor
  saturation: 0.2  # Saturation factor
  hue: 0.1        # Hue shift
```

## Best Practices

### 1. Use ImageNet Normalization for Pretrained Models

```yaml
normalize: true
normalize_mean: [0.485, 0.456, 0.406]
normalize_std: [0.229, 0.224, 0.225]
```

This is essential when using pretrained models (MobileNet, ResNet, etc.).

### 2. Disable Augmentation for Validation/Test

Create separate transforms:

```python
# Training (with augmentation)
train_transform = build_transform(..., is_training=True)

# Validation (no augmentation)
val_transform = build_transform(..., is_training=False)
```

### 3. Document Transform Settings in Config

Always specify transforms in your config file for reproducibility:

```yaml
# Good - Clear and reproducible
image_size: [224, 224]
normalize: true
normalize_mean: [0.485, 0.456, 0.406]
normalize_std: [0.229, 0.224, 0.225]
```

### 4. Match Transform to Model

Different models may expect different preprocessing:

```yaml
# For MobileNet/ResNet/EfficientNet (pretrained on ImageNet)
normalize_mean: [0.485, 0.456, 0.406]
normalize_std: [0.229, 0.224, 0.225]

# For custom models trained from scratch
normalize_mean: [0.5, 0.5, 0.5]
normalize_std: [0.5, 0.5, 0.5]

# For models without normalization
normalize: false
```

### 5. Consistent Image Sizes

Ensure image size matches model input requirements:

```yaml
# MobileNet, ResNet
image_size: [224, 224]

# EfficientNet-B0
image_size: [224, 224]

# EfficientNet-B7
image_size: [600, 600]
```

## Example Configurations

### Minimal (No Augmentation)

```yaml
image_size: [224, 224]
normalize: true
normalize_mean: [0.485, 0.456, 0.406]
normalize_std: [0.229, 0.224, 0.225]
augmentation: {}
```

### Standard Augmentation

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
  saturation: 0.2
  hue: 0.1
```

### Heavy Augmentation

```yaml
image_size: [224, 224]
normalize: true
normalize_mean: [0.485, 0.456, 0.406]
normalize_std: [0.229, 0.224, 0.225]
augmentation:
  random_horizontal_flip: true
  flip_prob: 0.5
  random_rotation: true
  rotation_degrees: 20
  random_crop: true
  crop_size: [200, 200]
  color_jitter: true
  brightness: 0.3
  contrast: 0.3
  saturation: 0.3
  hue: 0.2
```

### Custom Training (No Pretrained Normalization)

```yaml
image_size: [224, 224]
normalize: true
normalize_mean: [0.5, 0.5, 0.5]
normalize_std: [0.5, 0.5, 0.5]
augmentation:
  random_horizontal_flip: true
```

## Adding Custom Transforms

To add your own transform types, edit `src/li_4yp/utils/transforms.py`:

```python
def build_transform(...):
    # ...existing code...
    
    # Add your custom transform
    if augmentation.get("my_custom_transform", False):
        param = augmentation.get("my_param", default_value)
        transform_list.append(MyCustomTransform(param))
    
    # ...rest of code...
```

Then use in config:

```yaml
augmentation:
  my_custom_transform: true
  my_param: some_value
```

## Scikit-learn Models

For scikit-learn models that use extracted features (LAB values, etc.), transforms are typically not needed since you're working with numerical features directly, not images.

## Summary

| Use Case | Recommended Approach |
|----------|---------------------|
| Pretrained models | Config with ImageNet normalization |
| Custom PyTorch models | Config with custom normalization |
| Quick experiments | Presets (`"imagenet"`, `"basic"`) |
| Maximum control | Direct construction |
| Scikit-learn | No transforms (use features directly) |

Transform settings are automatically saved with your experiment configuration, ensuring complete reproducibility! 🎯
