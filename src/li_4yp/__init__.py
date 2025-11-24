"""Li 4YP Hair Swatch Prediction Package"""

from .models import (
    CNNRNNModel, MultiOutputCNN,
    DeeperShadeCNN, ShadeCNN,
    MbNetShadeCNN, ShadeEfficientNet,
    PixelStatNet, PixelMoreStatNet
)
from .data import HairSwatchDataset
from .training import compute_loss, multitask_loss, compute_class_weights, HierarchicalShadeLoss, ShadeEvaluator
from .experiments import Experiment, ExperimentConfig, ExperimentLogger, ModelRegistry

__version__ = "0.1.0"
__author__ = "Boting Li"

__all__ = [
    "CNNRNNModel",
    "MultiOutputCNN",
    "DeeperShadeCNN",
    "ShadeCNN",
    "MbNetShadeCNN",
    "ShadeEfficientNet",
    "PixelStatNet",
    "PixelMoreStatNet",
    "HairSwatchDataset",
    "compute_loss",
    "multitask_loss",
    "compute_class_weights",
    "HierarchicalShadeLoss",
    "ShadeEvaluator",
    "Experiment",
    "ExperimentConfig",
    "ExperimentLogger",
    "ModelRegistry"
]