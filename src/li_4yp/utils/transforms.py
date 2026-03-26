"""Transform utilities for data preprocessing."""

from typing import Optional, Dict, Any
import torch
from torchvision.transforms import v2

from .transform_presets import TRANSFORM_PRESETS

SUPPORTED_AUGMENTATION_KEYS = (
    "probability",
    "random_resized_crop",
    "resized_crop_scale",
    "resized_crop_ratio",
    "scale",
    "random_horizontal_flip",
    "random_vertical_flip",
    "flip_prob",
    "random_rotation",
    "rotation_degrees",
    "color_jitter",
    "brightness",
    "contrast",
    "saturation",
    "hue",
    "gaussian_blur",
    "blur_kernel_size",
    "blur_sigma",
)


def _validate_transform_inputs(
    image_size: tuple, augmentation: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """Validate and normalize transform config inputs."""
    if len(image_size) != 2:
        raise ValueError(
            f"image_size must contain exactly two values, got {image_size}"
        )

    if augmentation is None:
        return {}

    if not isinstance(augmentation, dict):
        raise TypeError(
            "augmentation must be a dictionary when provided, "
            f"got {type(augmentation)}"
        )

    unknown_keys = [
        key for key in augmentation if key not in SUPPORTED_AUGMENTATION_KEYS
    ]
    if unknown_keys:
        raise ValueError(
            f"Unsupported augmentation keys: {unknown_keys}. "
            f"Supported keys: {list(SUPPORTED_AUGMENTATION_KEYS)}"
        )

    normalized = dict(augmentation)
    if "scale" in normalized and "resized_crop_scale" not in normalized:
        normalized["resized_crop_scale"] = normalized["scale"]
    return normalized


def _build_post_process(
    normalize: bool, normalize_mean: tuple, normalize_std: tuple
) -> v2.Compose:
    """Build the shared post-processing path used after image transforms."""
    post_process = [v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]
    if normalize:
        post_process.append(v2.Normalize(mean=normalize_mean, std=normalize_std))
    return v2.Compose(post_process)


def _build_resize_only_path(image_size: tuple) -> v2.Compose:
    """Build the non-augmented image path."""
    return v2.Compose([v2.Resize(image_size)])


def _build_augmentation_path(
    image_size: tuple, augmentation: Dict[str, Any]
) -> v2.Compose:
    """Build the optional augmentation path for training."""
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
        aug_path.append(v2.RandomHorizontalFlip(p=augmentation.get("flip_prob", 0.5)))

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

    return v2.Compose(aug_path)


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
    augmentation = _validate_transform_inputs(image_size, augmentation)
    post_process = _build_post_process(normalize, normalize_mean, normalize_std)
    probability = augmentation.get("probability", 0.0)
    if probability < 0.0 or probability > 1.0:
        raise ValueError(
            f"Augmentation probability must be in [0.0, 1.0], got {probability}"
        )
    resize_only_path = _build_resize_only_path(image_size)

    if is_training:
        augmented_path = _build_augmentation_path(image_size, augmentation)
        select_transform = v2.RandomChoice(
            transforms=[augmented_path, resize_only_path],
            p=[probability, 1 - probability],
        )

        return v2.Compose([select_transform, post_process])

    return v2.Compose([resize_only_path, post_process])


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
