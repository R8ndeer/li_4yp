"""Predefined transform presets for the presentation training pipeline."""

import torch
from torchvision.transforms import v2


def _build_basic_preset(image_size: tuple) -> v2.Compose:
    """Build the shared resize/to-tensor preset without normalization."""
    return v2.Compose(
        [v2.Resize(image_size), v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]
    )


def _build_imagenet_preset(image_size: tuple) -> v2.Compose:
    """Build the shared ImageNet-style normalized preset."""
    return v2.Compose(
        [
            v2.Resize(image_size),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


TRANSFORM_PRESETS = {
    "imagenet": lambda size: _build_imagenet_preset(size),
    "basic": lambda size: _build_basic_preset(size),
    "mobilenet": lambda size: _build_imagenet_preset(size),
    "none": lambda size: _build_basic_preset(size),
}
