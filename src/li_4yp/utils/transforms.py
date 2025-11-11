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
    is_training: bool = True
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
    transform_list = []
    augmentation = augmentation or {}
    
    # Resize
    transform_list.append(v2.Resize(image_size))
    
    # Training augmentations
    if is_training:
        if augmentation.get("random_horizontal_flip", False):
            transform_list.append(
                v2.RandomHorizontalFlip(p=augmentation.get("flip_prob", 0.5))
            )
        
        if augmentation.get("random_rotation", False):
            degrees = augmentation.get("rotation_degrees", 10)
            transform_list.append(v2.RandomRotation(degrees))
        
        if augmentation.get("color_jitter", False):
            transform_list.append(
                v2.ColorJitter(
                    brightness=augmentation.get("brightness", 0.2),
                    contrast=augmentation.get("contrast", 0.2),
                    saturation=augmentation.get("saturation", 0.2),
                    hue=augmentation.get("hue", 0.1)
                )
            )
        
        if augmentation.get("random_crop", False):
            crop_size = augmentation.get("crop_size", image_size)
            transform_list.append(v2.RandomCrop(crop_size))
            transform_list.append(v2.Resize(image_size))  # Resize back
    
    # Convert to tensor
    transform_list.append(v2.ToImage())
    transform_list.append(v2.ToDtype(torch.float32, scale=True))
    
    # Normalization
    if normalize:
        transform_list.append(
            v2.Normalize(mean=normalize_mean, std=normalize_std)
        )
    
    return v2.Compose(transform_list)


def get_imagenet_transform(image_size: tuple = (224, 224)) -> v2.Compose:
    """Get standard ImageNet preprocessing transform.
    
    Args:
        image_size: Target image size
        
    Returns:
        ImageNet-style transform
    """
    return v2.Compose([
        v2.Resize(image_size),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def get_basic_transform(image_size: tuple = (224, 224)) -> v2.Compose:
    """Get basic transform without normalization.
    
    Args:
        image_size: Target image size
        
    Returns:
        Basic transform
    """
    return v2.Compose([
        v2.Resize(image_size),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True)
    ])


# Predefined transform presets
TRANSFORM_PRESETS = {
    "imagenet": lambda size: get_imagenet_transform(size),
    "basic": lambda size: get_basic_transform(size),
    "mobilenet": lambda size: v2.Compose([
        v2.Resize(size),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    "none": lambda size: v2.Compose([
        v2.Resize(size),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True)
    ])
}


def get_transform_from_preset(
    preset_name: str,
    image_size: tuple = (224, 224)
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
