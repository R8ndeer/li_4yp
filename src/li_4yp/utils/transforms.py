"""Transform utilities for data preprocessing."""

from typing import Optional, Dict, Any
import torch
from torchvision.transforms import v2


def build_transform(
    image_size: tuple = (224, 224),
    normalize: bool = True,
    normalize_mean: tuple = (0.485, 0.456, 0.406),
    normalize_std: tuple = (0.229, 0.224, 0.225),
    augmentation: Optional[Dict[str, Any]] = None,
    is_training: bool = True,
) -> v2.Compose:
    """Build a transform pipeline from configuration.

    Args:
        image_size: Target image size (height, width)
        normalize: Whether to normalize images
        normalize_mean: Mean values for normalization
        normalize_std: Std values for normalization
        augmentation: Dictionary of augmentation parameters
        is_training: Whether this is for training (applies augmentation)

    Returns:
        Composed transform pipeline
    """
    augmentation = augmentation or {}

    # TRANSFORMS SHARED BY TRAIN AND VAL
    post_process = [v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]
    if normalize:
        post_process.append(v2.Normalize(mean=normalize_mean, std=normalize_std))

    # TRAINING TRANSFORMS
    p = augmentation.get("probability", 0.0)
    if p < 0.0 or p > 1.0:
        raise ValueError(f"Augmentation probability must be in [0.0, 1.0], got {p}")
    orig_path = [
        v2.Resize(image_size),
    ]

    if is_training:
        aug_path = []

        if augmentation.get("random_resized_crop", False):
            aug_path.append(
                v2.RandomResizedCrop(
                    size=image_size,
                    scale=augmentation.get("resized_crop_scale", (0.8, 1.0)),
                    ratio=augmentation.get("resized_crop_ratio", (0.9, 1.1)),
                    interpolation=v2.InterpolationMode.BICUBIC,
                    antialias=True,
                )
            )

        if augmentation.get("random_horizontal_flip", False):
            aug_path.append(
                v2.RandomHorizontalFlip(p=augmentation.get("flip_prob", 0.5))
            )

        if augmentation.get("random_vertical_flip", False):
            aug_path.append(v2.RandomVerticalFlip(p=augmentation.get("flip_prob", 0.5)))

        if augmentation.get("random_rotation", False):
            aug_path.append(v2.RandomRotation(augmentation.get("rotation_degrees", 10)))

        if augmentation.get("color_jitter", False):
            aug_path.append(
                v2.ColorJitter(
                    brightness=augmentation.get("brightness", 0.1),
                    contrast=augmentation.get("contrast", 0.1),
                    saturation=augmentation.get("saturation", 0.1),
                    hue=augmentation.get("hue", 0.0),
                )
            )

        if augmentation.get("gaussian_blur", False):
            aug_path.append(
                v2.GaussianBlur(
                    kernel_size=augmentation.get("blur_kernel_size", 5),
                    sigma=augmentation.get("blur_sigma", (0.1, 2.0)),
                )
            )

        # Combine
        original_path = v2.Compose(orig_path)
        augmented_path = v2.Compose(aug_path)
        post_process = v2.Compose(post_process)

        select_transform = v2.RandomChoice(
            transforms=[original_path, augmented_path], p=[p, 1 - p]
        )

        return v2.Compose([select_transform, post_process])

    # VALIDATION TRANSFORMS
    else:
        return v2.Compose(orig_path + post_process)


def get_imagenet_transform(image_size: tuple = (224, 224)) -> v2.Compose:
    """Get standard ImageNet preprocessing transform.

    Args:
        image_size: Target image size

    Returns:
        ImageNet-style transform
    """
    return v2.Compose(
        [
            v2.Resize(image_size),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def get_basic_transform(image_size: tuple = (224, 224)) -> v2.Compose:
    """Get basic transform without normalization.

    Args:
        image_size: Target image size

    Returns:
        Basic transform
    """
    return v2.Compose(
        [v2.Resize(image_size), v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]
    )


# Predefined transform presets
TRANSFORM_PRESETS = {
    "imagenet": lambda size: get_imagenet_transform(size),
    "basic": lambda size: get_basic_transform(size),
    "mobilenet": lambda size: v2.Compose(
        [
            v2.Resize(size),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    ),
    "none": lambda size: v2.Compose(
        [v2.Resize(size), v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]
    ),
}


def get_transform_from_preset(
    preset_name: str, image_size: tuple = (224, 224)
) -> v2.Compose:
    """Get transform from preset name.

    Args:
        preset_name: Name of preset ("imagenet", "basic", "mobilenet", "none")
        image_size: Target image size

    Returns:
        Transform pipeline

    Raises:
        ValueError: If preset not found
    """
    if preset_name not in TRANSFORM_PRESETS:
        raise ValueError(
            f"Unknown transform preset: {preset_name}. "
            f"Available: {list(TRANSFORM_PRESETS.keys())}"
        )

    return TRANSFORM_PRESETS[preset_name](image_size)
