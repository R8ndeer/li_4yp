"""Utility exports for the presentation training pipeline."""

from importlib import import_module

__all__ = [
    "seed_all",
    "get_device",
    "build_transform",
    "get_imagenet_transform",
    "get_basic_transform",
    "get_transform_from_preset",
    "TRANSFORM_PRESETS",
]


_LAZY_EXPORTS = {
    "seed_all": ("li_4yp.utils.reproducibility", "seed_all"),
    "get_device": ("li_4yp.utils.reproducibility", "get_device"),
    "build_transform": ("li_4yp.utils.transforms", "build_transform"),
    "get_imagenet_transform": ("li_4yp.utils.transforms", "get_imagenet_transform"),
    "get_basic_transform": ("li_4yp.utils.transforms", "get_basic_transform"),
    "get_transform_from_preset": (
        "li_4yp.utils.transforms",
        "get_transform_from_preset",
    ),
    "TRANSFORM_PRESETS": ("li_4yp.utils.transforms", "TRANSFORM_PRESETS"),
}


def __getattr__(name: str):
    """Load utility helpers lazily to avoid importing torch on sklearn paths."""
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module 'li_4yp.utils' has no attribute '{name}'")

    module_name, attr_name = _LAZY_EXPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
