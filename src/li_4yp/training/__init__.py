"""Training module for hair swatch prediction models."""

from importlib import import_module

__all__ = [
    "compute_loss",
    "multitask_loss",
    "compute_class_weights",
    "HierarchicalShadeLoss",
    "ShadeEvaluator",
]


_LAZY_EXPORTS = {
    "compute_loss": ("li_4yp.training.losses", "compute_loss"),
    "multitask_loss": ("li_4yp.training.losses", "multitask_loss"),
    "compute_class_weights": ("li_4yp.training.losses", "compute_class_weights"),
    "HierarchicalShadeLoss": ("li_4yp.training.losses", "HierarchicalShadeLoss"),
    "ShadeEvaluator": ("li_4yp.training.metrics", "ShadeEvaluator"),
}


def __getattr__(name: str):
    """Load training helpers lazily so numpy-only paths remain available."""
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module 'li_4yp.training' has no attribute '{name}'")

    module_name, attr_name = _LAZY_EXPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
