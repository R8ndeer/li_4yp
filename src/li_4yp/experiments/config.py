"""Experiment configuration management."""

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional
from pathlib import Path
import yaml
import json
from datetime import datetime


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""
    
    # Experiment metadata
    name: str
    description: str = ""
    tags: list[str] = field(default_factory=list)
    
    # Model configuration
    model_type: str = "pytorch"  # "pytorch" or "sklearn"
    model_name: str = ""
    model_params: Dict[str, Any] = field(default_factory=dict)
    
    # Data configuration
    dataset_name: str = "masterlist_v6"
    data_dir: str = "data/masterlist_v6"
    csv_file: str = "masterlist_v6_full_features.csv"
    train_split: float = 0.8
    random_seed: int = 42
    
    # Training configuration (PyTorch)
    batch_size: int = 32
    num_epochs: int = 100
    learning_rate: float = 1e-3
    optimizer: str = "Adam"
    optimizer_params: Dict[str, Any] = field(default_factory=dict)
    scheduler: Optional[str] = None
    scheduler_params: Dict[str, Any] = field(default_factory=dict)
    early_stopping_patience: int = 10
    
    # Loss configuration
    loss_function: str = "CrossEntropyLoss"
    loss_weights: tuple = (1.0, 1.0, 1.0)  # for multi-task losses
    
    # Evaluation configuration
    eval_metrics: list[str] = field(default_factory=lambda: [
        "base_acc",
        "tol_base_acc", 
        "primary_acc",
        "secondary_acc",
        "Hierarchical Score",
        "Exact Match",
        "Base-Primary Exact Match"
    ])
    evaluator_params: Dict[str, Any] = field(default_factory=lambda: {
        "eos_token": -1,
        "weights": (0.3, 0.1),
        "tol_base": 4
    })
    
    # Output configuration
    save_dir: str = "experiments"
    save_model: bool = True
    save_predictions: bool = True
    save_history: bool = True
    
    # Reproducibility
    device: str = "auto"  # "auto", "cpu", "cuda"
    num_workers: int = 0
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.model_type not in ["pytorch", "sklearn"]:
            raise ValueError(f"model_type must be 'pytorch' or 'sklearn', got {self.model_type}")
        
        if self.train_split <= 0 or self.train_split >= 1:
            raise ValueError(f"train_split must be between 0 and 1, got {self.train_split}")
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ExperimentConfig":
        """Create config from dictionary."""
        return cls(**config_dict)
    
    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "ExperimentConfig":
        """Load config from YAML file."""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_json(cls, json_path: str | Path) -> "ExperimentConfig":
        """Load config from JSON file."""
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)
    
    def to_yaml(self, yaml_path: str | Path) -> None:
        """Save config to YAML file."""
        Path(yaml_path).parent.mkdir(parents=True, exist_ok=True)
        with open(yaml_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
    
    def to_json(self, json_path: str | Path) -> None:
        """Save config to JSON file."""
        Path(json_path).parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def get_experiment_dir(self) -> Path:
        """Get the experiment directory path."""
        # Create timestamp-based directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = f"{timestamp}_{self.name}"
        return Path(self.save_dir) / exp_name
