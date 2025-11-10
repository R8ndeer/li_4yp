"""Li 4YP Hair Swatch Prediction Package"""

from .models import CNNRNNModel, MultiOutputCNN
from .data import HairSwatchDataset
from .training import compute_loss, multitask_loss, ShadeEvaluator
from .experiments import Experiment, ExperimentConfig, ExperimentLogger, ModelRegistry

__version__ = "0.1.0"
__author__ = "Boting Li"

__all__ = [
    "CNNRNNModel",
    "MultiOutputCNN",
    "HairSwatchDataset",
    "compute_loss",
    "multitask_loss",
    "ShadeEvaluator",
    "Experiment",
    "ExperimentConfig",
    "ExperimentLogger",
    "ModelRegistry"
]