"""Experiment tracking and management exports."""

from importlib import import_module

__all__ = ["Experiment", "ExperimentLogger", "ExperimentConfig", "ModelRegistry"]


_LAZY_EXPORTS = {
    "Experiment": ("li_4yp.experiments.experiment", "Experiment"),
    "ExperimentLogger": ("li_4yp.experiments.logger", "ExperimentLogger"),
    "ExperimentConfig": ("li_4yp.experiments.config", "ExperimentConfig"),
    "ModelRegistry": ("li_4yp.experiments.model_registry", "ModelRegistry"),
}


def __getattr__(name: str):
    """Load experiment submodules lazily to avoid eager ML imports."""
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module 'li_4yp.experiments' has no attribute '{name}'")

    module_name, attr_name = _LAZY_EXPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
