"""Experiment configuration management."""

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional
from pathlib import Path
import yaml
import json
from datetime import datetime

ALLOWED_MODEL_TYPES = ("pytorch", "sklearn")
SHARED_REQUIRED_FIELDS = (
    "name",
    "model_type",
    "model_name",
    "data_dir",
    "csv_file",
    "train_split",
    "save_dir",
)
PYTORCH_REQUIRED_FIELDS = (
    "dataset_class",
    "batch_size",
    "num_epochs",
    "learning_rate",
    "optimizer",
    "loss_name",
    "image_size",
    "normalize",
    "device",
    "num_workers",
)
SKLEARN_REQUIRED_FIELDS = ("feature_cols", "label_cols")


def _is_missing(value: Any) -> bool:
    """Return True when a config field is absent or empty."""
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    return False


def _format_missing_fields_error(
    model_type: Any,
    shared_missing: list[str],
    family_missing: list[str],
) -> str:
    """Build a grouped error message for required config fields."""
    lines = ["Config is missing required fields."]
    if shared_missing:
        lines.append(f"Shared required fields: {', '.join(shared_missing)}")
    if family_missing and model_type == "pytorch":
        lines.append(f"PyTorch required fields: {', '.join(family_missing)}")
    if family_missing and model_type == "sklearn":
        lines.append(f"sklearn required fields: {', '.join(family_missing)}")
    return "\n".join(lines)


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""

    # Experiment metadata
    name: Optional[str] = None
    description: str = ""
    tags: list[str] = field(default_factory=list)

    # Model configuration
    model_type: Optional[str] = None
    model_name: Optional[str] = None
    model_params: Dict[str, Any] = field(default_factory=dict)

    # Data configuration
    dataset_class: Optional[str] = None
    dataset_params: Dict[str, Any] = field(default_factory=dict)
    dataset_name: str = ""
    data_dir: Optional[str] = None
    csv_file: Optional[str] = None
    feature_cols: Optional[list[str]] = None
    label_cols: Optional[list[str]] = None
    train_split: Optional[float] = None
    val_fold_idx: Optional[int] = None  # For fixed-fold CV
    random_seed: int = 42

    # Training configuration (PyTorch)
    batch_size: Optional[int] = None
    num_epochs: Optional[int] = None
    learning_rate: Optional[float] = None
    optimizer: Optional[str] = None
    optimizer_params: Dict[str, Any] = field(default_factory=dict)
    scheduler: Optional[str] = None
    scheduler_params: Dict[str, Any] = field(default_factory=dict)
    early_stopping_patience: int = 10

    # Loss configuration
    loss_function: str = "CrossEntropyLoss"
    loss_weights: tuple = (1.0, 1.0, 1.0)  # for multi-task losses
    loss_name: str = ""
    loss_params: Dict[str, Any] = field(default_factory=dict)

    # Evaluation configuration
    eval_metrics: list[str] = field(
        default_factory=lambda: [
            "base_acc",
            "tol_base_acc",
            "primary_acc",
            "secondary_acc",
            "Hierarchical Score",
            "Exact Match",
            "Base-Primary Exact Match",
            "Primary-Secondary Exact Match",
        ]
    )
    evaluator_params: Dict[str, Any] = field(
        default_factory=lambda: {"eos_token": -1, "weights": (0.3, 0.1), "tol_base": 4}
    )

    # Output configuration
    save_dir: Optional[str] = None
    save_model: bool = True
    save_predictions: bool = True
    save_history: bool = True

    # Data augmentation & transforms (PyTorch)
    use_transform_preset: bool = True
    transform_preset: Optional[str] = "imagenet"
    image_size: Optional[tuple] = None
    normalize: Optional[bool] = None
    normalize_mean: tuple = (0.485, 0.456, 0.406)  # ImageNet defaults
    normalize_std: tuple = (0.229, 0.224, 0.225)
    augmentation: Dict[str, Any] = field(
        default_factory=dict
    )  # e.g., {"random_flip": True}

    # Reproducibility
    device: Optional[str] = None
    num_workers: Optional[int] = None

    # Config dump
    _PYTORCH_FIELDS = {
        "dataset_class",
        "dataset_params",
        "dataset_name",
        "batch_size",
        "num_epochs",
        "learning_rate",
        "optimizer",
        "optimizer_params",
        "scheduler",
        "scheduler_params",
        "early_stopping_patience",
        "loss_function",
        "loss_weights",
        "loss_name",
        "loss_params",
        "use_transform_preset",
        "transform_preset",
        "image_size",
        "normalize",
        "normalize_mean",
        "normalize_std",
        "augmentation",
        "device",
        "num_workers",
        "val_fold_idx",
    }
    _SKLEARN_FIELDS = {"feature_cols", "label_cols"}

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.model_type not in ALLOWED_MODEL_TYPES:
            raise ValueError(
                f"model_type must be one of {list(ALLOWED_MODEL_TYPES)}, got {self.model_type}"
            )

        if _is_missing(self.name):
            raise ValueError("name must be a non-empty string.")
        if _is_missing(self.model_name):
            raise ValueError("model_name must be a non-empty string.")
        if _is_missing(self.data_dir):
            raise ValueError("data_dir must be a non-empty string.")
        if _is_missing(self.csv_file):
            raise ValueError("csv_file must be a non-empty string.")
        if _is_missing(self.save_dir):
            raise ValueError("save_dir must be a non-empty string.")
        if self.train_split is None:
            raise ValueError("train_split must be provided.")
        if self.train_split <= 0 or self.train_split >= 1:
            raise ValueError(
                f"train_split must be between 0 and 1, got {self.train_split}"
            )

        if self.val_fold_idx is not None and not isinstance(self.val_fold_idx, int):
            raise TypeError(
                f"val_fold_idx must be an integer when provided, got {type(self.val_fold_idx)}"
            )

        if self.model_type == "pytorch":
            if _is_missing(self.dataset_class):
                raise ValueError("dataset_class must be provided for PyTorch configs.")
            if self.batch_size is None:
                raise ValueError("batch_size must be provided for PyTorch configs.")
            if self.num_epochs is None:
                raise ValueError("num_epochs must be provided for PyTorch configs.")
            if self.learning_rate is None:
                raise ValueError("learning_rate must be provided for PyTorch configs.")
            if _is_missing(self.optimizer):
                raise ValueError("optimizer must be provided for PyTorch configs.")
            if _is_missing(self.loss_name):
                raise ValueError("loss_name must be provided for PyTorch configs.")
            if self.image_size is None:
                raise ValueError("image_size must be provided for PyTorch configs.")
            if len(self.image_size) != 2:
                raise ValueError(
                    f"image_size must contain exactly two values, got {self.image_size}"
                )
            if self.normalize is None:
                raise ValueError("normalize must be provided for PyTorch configs.")
            if _is_missing(self.device):
                raise ValueError("device must be provided for PyTorch configs.")
            if self.num_workers is None:
                raise ValueError("num_workers must be provided for PyTorch configs.")
            if self.batch_size <= 0:
                raise ValueError(f"batch_size must be positive, got {self.batch_size}")
            if self.num_epochs <= 0:
                raise ValueError(f"num_epochs must be positive, got {self.num_epochs}")
            if self.learning_rate <= 0:
                raise ValueError(
                    f"learning_rate must be positive, got {self.learning_rate}"
                )
            if self.early_stopping_patience <= 0:
                raise ValueError(
                    "early_stopping_patience must be positive, "
                    f"got {self.early_stopping_patience}"
                )
            if self.use_transform_preset and _is_missing(self.transform_preset):
                raise ValueError(
                    "transform_preset must be provided when use_transform_preset is true."
                )
            if self.use_transform_preset and self.augmentation:
                raise ValueError(
                    "Cannot use both transform preset and custom augmentations. "
                    "Set use_transform_preset to false to use augmentation settings."
                )
        elif self.model_type == "sklearn":
            if not self.feature_cols:
                raise ValueError("feature_cols must be provided for sklearn configs.")
            if not self.label_cols:
                raise ValueError("label_cols must be provided for sklearn configs.")

        if self.num_workers is not None and self.num_workers < 0:
            raise ValueError(f"num_workers cannot be negative, got {self.num_workers}")

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ExperimentConfig":
        """Create config from dictionary."""
        if config_dict is None:
            raise ValueError("Config file is empty.")
        if not isinstance(config_dict, dict):
            raise TypeError(
                f"Config must load to a dictionary, got {type(config_dict)}."
            )

        model_type = config_dict.get("model_type")
        shared_missing = [
            field
            for field in SHARED_REQUIRED_FIELDS
            if _is_missing(config_dict.get(field))
        ]

        family_missing: list[str] = []
        if not _is_missing(model_type) and model_type in ALLOWED_MODEL_TYPES:
            if model_type == "pytorch":
                family_missing = [
                    field
                    for field in PYTORCH_REQUIRED_FIELDS
                    if _is_missing(config_dict.get(field))
                ]
                if config_dict.get("use_transform_preset", True) and _is_missing(
                    config_dict.get("transform_preset")
                ):
                    family_missing.append("transform_preset")
            elif model_type == "sklearn":
                family_missing = [
                    field
                    for field in SKLEARN_REQUIRED_FIELDS
                    if _is_missing(config_dict.get(field))
                ]

        if shared_missing or family_missing:
            raise ValueError(
                _format_missing_fields_error(model_type, shared_missing, family_missing)
            )

        if not _is_missing(model_type) and model_type not in ALLOWED_MODEL_TYPES:
            raise ValueError(
                f"Unsupported model_type '{model_type}'. "
                f"Expected one of: {list(ALLOWED_MODEL_TYPES)}."
            )

        try:
            return cls(**config_dict)
        except TypeError as e:
            raise TypeError(f"Invalid config fields: {e}") from e

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "ExperimentConfig":
        """Load config from YAML file."""
        with open(yaml_path, "r") as f:
            config_dict = yaml.safe_load(f)
        try:
            return cls.from_dict(config_dict)
        except Exception as e:
            raise type(e)(f"Invalid config '{yaml_path}': {e}") from e

    @classmethod
    def from_json(cls, json_path: str | Path) -> "ExperimentConfig":
        """Load config from JSON file."""
        with open(json_path, "r") as f:
            config_dict = json.load(f)
        try:
            return cls.from_dict(config_dict)
        except Exception as e:
            raise type(e)(f"Invalid config '{json_path}': {e}") from e

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        config_dict = asdict(self)
        if self.model_type == "pytorch":
            for f in self._SKLEARN_FIELDS:
                config_dict.pop(f, None)
        elif self.model_type == "sklearn":
            for f in self._PYTORCH_FIELDS:
                config_dict.pop(f, None)
        return config_dict

    def to_yaml(self, yaml_path: str | Path) -> None:
        """Save config to YAML file."""
        Path(yaml_path).parent.mkdir(parents=True, exist_ok=True)
        with open(yaml_path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    def to_json(self, json_path: str | Path) -> None:
        """Save config to JSON file."""
        Path(json_path).parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def get_experiment_dir(self) -> Path:
        """Get the experiment directory path."""
        # timestamp-based directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = f"{timestamp}_{self.name}"
        return Path(self.save_dir) / exp_name
