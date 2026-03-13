# Scripts

This directory contains entry point scripts for various tasks in the hair swatch prediction project.

## Available Scripts

### `preprocess_data.py`
Preprocesses the hair swatch dataset by extracting shade codes from image filenames.

```bash
python scripts/preprocess_data.py
```

Features:
- Parses shade codes from image filenames 
- Handles various filename formats (X.XXX, base-only, etc.)
- Filters out "half" bases (containing "1_2")
- Generates CSV with shade code labels

### `train.py`
Main training script for hair swatch shade prediction models.

```bash
# Train CNN-RNN model (default)
python scripts/train.py --model cnn_rnn

# Train Multi-Output CNN model
python scripts/train.py --model multi_output_cnn

# Use custom config file
python scripts/train.py --model cnn_rnn --config config/custom.yaml
```

Features:
- Supports multiple model architectures
- Automatic best model saving
- Configurable training parameters
- Progress tracking and metrics

## Usage Examples

```bash
# Full pipeline
python scripts/preprocess_data.py
python scripts/train.py --model cnn_rnn

# Custom training
python scripts/train.py --model multi_output_cnn --config config/experiment.yaml
```

## Requirements

Make sure you have installed the package in development mode:

```bash
pip install -e .
```

Or add the project to your Python path by running scripts from the project root.
