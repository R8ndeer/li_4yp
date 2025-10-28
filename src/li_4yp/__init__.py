"""Li 4YP Hair Swatch Prediction Package"""

from .models import CNNRNNModel, MultiOutputCNN
from .data import HairSwatchDataset
from .training import compute_loss, ShadeEvaluator

__version__ = "0.1.0"
__author__ = "Boting Li"

__all__ = [
    "CNNRNNModel",
    "MultiOutputCNN", 
    "HairSwatchDataset",
    "compute_loss",
    "ShadeEvaluator"
]
