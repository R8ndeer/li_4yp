"""Li 4YP Hair Swatch Prediction Package."""

from importlib import import_module

__version__ = "0.1.0"
__author__ = "Boting Li"

__all__ = [
    "AttentivePixelStatNet",
    "AttentiveStatNetOneMoment",
    "HairSwatchDataset",
    "DigitalSwatchDataset",
    "HybridSwatchDataset",
    "compute_loss",
    "multitask_loss",
    "compute_class_weights",
    "HierarchicalShadeLoss",
    "ShadeEvaluator",
    "Experiment",
    "ExperimentConfig",
    "ExperimentLogger",
    "ModelRegistry",
]


_LAZY_EXPORTS = {
    "AttentivePixelStatNet": ("li_4yp.models", "AttentivePixelStatNet"),
    "AttentiveStatNetOneMoment": ("li_4yp.models", "AttentiveStatNetOneMoment"),
    "HairSwatchDataset": ("li_4yp.data", "HairSwatchDataset"),
    "DigitalSwatchDataset": ("li_4yp.data", "DigitalSwatchDataset"),
    "HybridSwatchDataset": ("li_4yp.data", "HybridSwatchDataset"),
    "compute_loss": ("li_4yp.training", "compute_loss"),
    "multitask_loss": ("li_4yp.training", "multitask_loss"),
    "compute_class_weights": ("li_4yp.training", "compute_class_weights"),
    "HierarchicalShadeLoss": ("li_4yp.training", "HierarchicalShadeLoss"),
    "ShadeEvaluator": ("li_4yp.training", "ShadeEvaluator"),
    "Experiment": ("li_4yp.experiments", "Experiment"),
    "ExperimentConfig": ("li_4yp.experiments", "ExperimentConfig"),
    "ExperimentLogger": ("li_4yp.experiments", "ExperimentLogger"),
    "ModelRegistry": ("li_4yp.experiments", "ModelRegistry"),
}


def __getattr__(name: str):
    """Load heavy submodules only when their exports are requested."""
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module 'li_4yp' has no attribute '{name}'")

    module_name, attr_name = _LAZY_EXPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
