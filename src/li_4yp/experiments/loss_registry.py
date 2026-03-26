from typing import Callable, Dict, Optional
import torch.nn as nn

from ..training.losses import HierarchicalShadeLoss


class LossRegistry:
    """Registry for loss functions."""

    _registry: Dict[str, Callable] = {}
    _descriptions: Dict[str, str] = {}

    @classmethod
    def register(cls, name: str, constructor: Callable, description: str = "") -> None:
        """Register a loss function.

        Args:
            name: Unique loss function name
            constructor: Callable that returns a loss function instance
            description: Description of the loss function
        """
        if cls.is_registered(name):
            raise ValueError(f"Loss function '{name}' is already registered")

        cls._registry[name] = constructor
        cls._descriptions[name] = description

    @classmethod
    def get(cls, name: str, **kwargs) -> Optional[Callable]:
        """Get a registered loss function.

        Args:
            name: Loss function name
            **kwargs: Arguments to pass to the loss constructor

        Returns:
            Loss function instance
        """
        if not cls.is_registered(name):
            raise ValueError(
                f"Loss function '{name}' not found. Available losses: {list(cls._registry.keys())}"
            )

        return cls._registry[name](**kwargs)

    @classmethod
    def list_losses(cls) -> Dict[str, str]:
        """List all registered loss functions with descriptions.

        Returns:
            Dictionary mapping loss function names to descriptions
        """
        return cls._descriptions.copy()

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check if a loss function is registered.

        Args:
            name (str): Loss function name

        Returns:
            bool: True if registered, False otherwise
        """
        return name in cls._registry

    @classmethod
    def unregister(cls, name: str) -> None:
        """Unregister a loss function.

        Args:
            name (str): Loss function name
        """
        if not cls.is_registered(name):
            raise ValueError(f"Loss function '{name}' not found")

        del cls._registry[name]
        del cls._descriptions[name]

    @classmethod
    def clear(cls) -> None:
        """Clear all registered loss functions."""
        cls._registry.clear()
        cls._descriptions.clear()

    @classmethod
    def reregister(
        cls, name: str, constructor: Callable, description: str = ""
    ) -> None:
        """Reregister (overwrite) a loss function.

        Args:
            name (str): Loss function name
            constructor (Callable): New constructor for the loss function
            description (str): New description for the loss function
        """
        if cls.is_registered(name):
            cls.unregister(name)
        cls.register(name, constructor, description)


def register_default_losses() -> None:
    """Register default loss functions."""
    if LossRegistry.is_registered("CrossEntropyLoss"):
        return

    LossRegistry.register(
        "CrossEntropyLoss",
        lambda **kwargs: nn.CrossEntropyLoss(**kwargs),
        "Standard Cross Entropy Loss",
    )

    if LossRegistry.is_registered("MSELoss"):
        return

    LossRegistry.register(
        "MSELoss", lambda **kwargs: nn.MSELoss(**kwargs), "Mean Squared Error Loss"
    )

    if LossRegistry.is_registered("L1Loss"):
        return

    LossRegistry.register(
        "L1Loss", lambda **kwargs: nn.L1Loss(**kwargs), "Mean Absolute Error Loss"
    )

    if LossRegistry.is_registered("HierarchicalShadeLoss"):
        return

    LossRegistry.register(
        "HierarchicalShadeLoss",
        lambda **kwargs: HierarchicalShadeLoss(**kwargs),
        "Custom Hierarchical Shade Loss with class weights for color shade prediction",
    )
