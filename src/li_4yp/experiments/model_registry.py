"""Model registry for the presentation training slice."""

from typing import Any, Callable, Dict, Optional


class ModelRegistry:
    """Registry for managing model constructors."""

    _registry: Dict[str, Callable] = {}
    _descriptions: Dict[str, str] = {}

    @classmethod
    def register(cls, name: str, constructor: Callable, description: str = "") -> None:
        """Register a model constructor.

        Args:
            name: Unique model name
            constructor: Callable that returns a model instance
            description: Description of the model
        """
        if cls.is_registered(name):
            raise ValueError(f"Model '{name}' is already registered")

        cls._registry[name] = constructor
        cls._descriptions[name] = description

    @classmethod
    def get(cls, name: str, **kwargs) -> Any:
        """Get a model instance.

        Args:
            name: Model name
            **kwargs: Arguments to pass to model constructor

        Returns:
            Model instance
        """
        if not cls.is_registered(name):
            raise ValueError(
                f"Model '{name}' not found. Available models: {list(cls._registry.keys())}"
            )

        return cls._registry[name](**kwargs)

    @classmethod
    def list_models(cls) -> Dict[str, str]:
        """List all registered models with descriptions.

        Returns:
            Dictionary mapping model names to descriptions
        """
        return cls._descriptions.copy()

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check if a model is registered.

        Args:
            name: Model name

        Returns:
            True if model is registered
        """
        return name in cls._registry

    @classmethod
    def unregister(cls, name: str) -> None:
        """Unregister a model.

        Args:
            name: Model name to unregister

        Raises:
            ValueError: If model is not registered
        """
        if not cls.is_registered(name):
            raise ValueError(f"Model '{name}' is not registered")

        del cls._registry[name]
        del cls._descriptions[name]

    @classmethod
    def clear(cls) -> None:
        """Clear all registered models.

        Warning: This will remove all models including built-in ones.
        Use with caution!
        """
        cls._registry.clear()
        cls._descriptions.clear()

    @classmethod
    def reregister(
        cls, name: str, constructor: Callable, description: str = ""
    ) -> None:
        """Re-register a model (overwrites existing registration).

        Args:
            name: Model name
            constructor: New constructor function
            description: New description
        """
        if cls.is_registered(name):
            cls.unregister(name)
        cls.register(name, constructor, description)


def register_pytorch_models():
    """Register the PyTorch models kept in the presentation slice."""
    from li_4yp.models import AttentivePixelStatNet, AttentiveStatNetOneMoment

    if ModelRegistry.is_registered("attentive_pixel_stat_net"):
        return

    ModelRegistry.register(
        name="attentive_pixel_stat_net",
        constructor=lambda **kwargs: AttentivePixelStatNet(
            num_classes=kwargs.get("num_classes", [12, 11, 11]),
            dropout_rate=kwargs.get("dropout_rate", 0.2),
        ),
        description="AttentivePixelStatNet: An attention-based Deep Sets model for hair color classification.",
    )

    if ModelRegistry.is_registered("attentive_stat_net_one_moment"):
        return

    ModelRegistry.register(
        name="attentive_stat_net_one_moment",
        constructor=lambda **kwargs: AttentiveStatNetOneMoment(
            num_classes=kwargs.get("num_classes", [12, 11, 11]),
            dropout_rate=kwargs.get("dropout_rate", 0.2),
        ),
        description="AttentiveStatNetOneMoment: Simplified attention-based model using only weighted mean.",
    )


def register_sklearn_models():
    """Register the scikit-learn models supported by the original repo."""

    from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
    from sklearn.multioutput import MultiOutputClassifier
    from sklearn.neural_network import MLPClassifier

    if ModelRegistry.is_registered("multi_output_random_forest"):
        return

    ModelRegistry.register(
        name="multi_output_random_forest",
        constructor=lambda **kwargs: MultiOutputClassifier(
            RandomForestClassifier(
                n_estimators=kwargs.get("n_estimators", 200),
                random_state=kwargs.get("random_state", 42),
                n_jobs=kwargs.get("n_jobs", -1),
                **{
                    k: v
                    for k, v in kwargs.items()
                    if k not in ["n_estimators", "random_state", "n_jobs"]
                },
            )
        ),
        description="Random Forest multi-output classifier",
    )

    ModelRegistry.register(
        name="multi_output_extra_trees",
        constructor=lambda **kwargs: MultiOutputClassifier(
            ExtraTreesClassifier(
                n_estimators=kwargs.get("n_estimators", 200),
                random_state=kwargs.get("random_state", 42),
                n_jobs=kwargs.get("n_jobs", -1),
                **{
                    k: v
                    for k, v in kwargs.items()
                    if k not in ["n_estimators", "random_state", "n_jobs"]
                },
            )
        ),
        description="Extra Trees multi-output classifier",
    )

    ModelRegistry.register(
        name="multi_output_mlp",
        constructor=lambda **kwargs: MultiOutputClassifier(
            MLPClassifier(
                hidden_layer_sizes=kwargs.get("hidden_layer_sizes", (256, 128)),
                max_iter=kwargs.get("max_iter", 500),
                random_state=kwargs.get("random_state", 42),
                **{
                    k: v
                    for k, v in kwargs.items()
                    if k not in ["hidden_layer_sizes", "max_iter", "random_state"]
                },
            )
        ),
        description="Multi-layer Perceptron multi-output classifier",
    )


def register_custom_model(
    name: str,
    model_class: type,
    default_params: Optional[Dict[str, Any]] = None,
    description: str = "",
):
    """Register a custom model.

    Args:
        name: Model name
        model_class: Model class
        default_params: Default parameters for the model
        description: Model description

    Example:
        >>> class MyModel(nn.Module):
        ...     def __init__(self, hidden_dim=128):
        ...         super().__init__()
        ...         self.fc = nn.Linear(10, hidden_dim)
        >>>
        >>> register_custom_model(
        ...     name="my_model",
        ...     model_class=MyModel,
        ...     default_params={"hidden_dim": 256},
        ...     description="My custom model"
        ... )
    """
    default_params = default_params or {}

    def constructor(**kwargs):
        params = {**default_params, **kwargs}
        return model_class(**params)

    ModelRegistry.register(name, constructor, description)
