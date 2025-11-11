"""Model registry for easy model instantiation."""

from typing import Any, Callable, Dict, Optional
import torch
import torch.nn as nn


class ModelRegistry:
    """Registry for managing model constructors."""
    
    _registry: Dict[str, Callable] = {}
    _descriptions: Dict[str, str] = {}
    
    @classmethod
    def register(
        cls,
        name: str,
        constructor: Callable,
        description: str = ""
    ) -> None:
        """Register a model constructor.
        
        Args:
            name: Unique model name
            constructor: Callable that returns a model instance
            description: Description of the model
        """
        if name in cls._registry:
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
        if name not in cls._registry:
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
        if name not in cls._registry:
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
    def reregister(cls, name: str, constructor: Callable, description: str = "") -> None:
        """Re-register a model (overwrites existing registration).
        
        Args:
            name: Model name
            constructor: New constructor function
            description: New description
        """
        if name in cls._registry:
            cls.unregister(name)
        cls.register(name, constructor, description)


def register_pytorch_models():
    """Register built-in PyTorch models."""
    from li_4yp.models import CNNRNNModel, MultiOutputCNN, ShadeCNN
    
    # CNN-RNN Model
    ModelRegistry.register(
        name="cnn_rnn",
        constructor=lambda **kwargs: CNNRNNModel(
            vocab_size=kwargs.get('vocab_size', 12),
            embed_dim=kwargs.get('embed_dim', 64),
            hidden_dim=kwargs.get('hidden_dim', 256),
            output_dim=kwargs.get('output_dim', 12),
            feat_dim=kwargs.get('feat_dim', 128)
        ),
        description="CNN-RNN model for sequential shade prediction"
    )
    
    # Multi-Output CNN
    ModelRegistry.register(
        name="multi_output_cnn",
        constructor=lambda **kwargs: MultiOutputCNN(
            vocab_size=kwargs.get('vocab_size', 12)
        ),
        description="Multi-output CNN for parallel shade component prediction"
    )

    # ShadeCNN
    ModelRegistry.register(
        name="shade_cnn",
        constructor=lambda **kwargs: ShadeCNN(
            num_classes=kwargs.get('num_classes', [12, 11, 11])
        ),
        description="A simple CNN for multi-task hair color classification"
    )


def register_sklearn_models():
    """Register scikit-learn models."""
    from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.multioutput import MultiOutputClassifier
    
    # Random Forest
    ModelRegistry.register(
        name="multi_output_random_forest",
        constructor=lambda **kwargs: MultiOutputClassifier(
            RandomForestClassifier(
                n_estimators=kwargs.get('n_estimators', 200),
                random_state=kwargs.get('random_state', 42),
                n_jobs=kwargs.get('n_jobs', -1),
                **{k: v for k, v in kwargs.items() 
                   if k not in ['n_estimators', 'random_state', 'n_jobs']}
            )
        ),
        description="Random Forest multi-output classifier"
    )
    
    # Extra Trees
    ModelRegistry.register(
        name="multi_output_extra_trees",
        constructor=lambda **kwargs: MultiOutputClassifier(
            ExtraTreesClassifier(
                n_estimators=kwargs.get('n_estimators', 200),
                random_state=kwargs.get('random_state', 42),
                n_jobs=kwargs.get('n_jobs', -1),
                **{k: v for k, v in kwargs.items() 
                   if k not in ['n_estimators', 'random_state', 'n_jobs']}
            )
        ),
        description="Extra Trees multi-output classifier"
    )
    
    # MLP
    ModelRegistry.register(
        name="multi_output_mlp",
        constructor=lambda **kwargs: MultiOutputClassifier(
            MLPClassifier(
                hidden_layer_sizes=kwargs.get('hidden_layer_sizes', (256, 128)),
                max_iter=kwargs.get('max_iter', 500),
                random_state=kwargs.get('random_state', 42),
                **{k: v for k, v in kwargs.items() 
                   if k not in ['hidden_layer_sizes', 'max_iter', 'random_state']}
            )
        ),
        description="Multi-layer Perceptron multi-output classifier"
    )


def register_custom_model(
    name: str,
    model_class: type,
    default_params: Optional[Dict[str, Any]] = None,
    description: str = ""
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


# Auto-register built-in models
register_pytorch_models()
register_sklearn_models()
