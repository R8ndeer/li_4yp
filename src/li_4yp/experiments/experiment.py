from __future__ import annotations

"""Main Experiment class for running and tracking experiments."""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import pickle
import time

from .config import ExperimentConfig
from .logger import ExperimentLogger
from .model_registry import ModelRegistry
from .loss_registry import LossRegistry
from li_4yp.training import ShadeEvaluator


class Experiment:
    """Main experiment class for training and evaluation."""

    def __init__(self, config: ExperimentConfig):
        """Initialize experiment.

        Args:
            config: Experiment configuration
        """
        self.config = config

        # Set random seed
        if config.model_type == "pytorch":
            self._require_torch()
            from li_4yp.utils import seed_all

            seed_all(config.random_seed)

        # Create experiment directory
        self.exp_dir = config.get_experiment_dir()
        self.exp_dir.mkdir(parents=True, exist_ok=True)

        # Initialize logger
        self.logger = ExperimentLogger(
            log_dir=self.exp_dir, experiment_name=config.name
        )

        # Log configuration
        self.logger.log_config(config.to_dict())

        # Initialize components
        self.model = None
        self.optimizer = None
        self.loss_fn = None
        self.scheduler = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.evaluator = None
        self.device = self._get_device()

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float("inf")
        self.epochs_without_improvement = 0

        self.logger.info(f"Experiment directory: {self.exp_dir}")
        self.logger.info(f"Device: {self.device}")

    def _get_device(self) -> torch.device:
        """Get compute device."""
        if self.config.model_type != "pytorch":
            return "cpu"

        torch, _, _, _, _ = self._require_torch()
        from li_4yp.utils import get_device

        if self.config.device == "auto":
            device = get_device()
        else:
            device = torch.device(self.config.device)
        return device

    @staticmethod
    def _require_torch():
        """Import torch only when a PyTorch-specific path is used."""
        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader, Subset, random_split
        except Exception as e:  # pragma: no cover - depends on local ML runtime
            raise RuntimeError(
                "PyTorch runtime is unavailable in this environment."
            ) from e

        return torch, nn, DataLoader, Subset, random_split

    def _build_train_transform(self):
        """Build the training transform defined by the config."""
        self._require_torch()
        from li_4yp.utils import build_transform, get_transform_from_preset

        if self.config.use_transform_preset:
            if self.config.augmentation:
                msg = "Cannot use both transform preset and custom augmentations."
                msg += (
                    " Set 'use_transform_preset' to false EXPLICITLY to use custom "
                    "augmentations."
                )
                raise ValueError(msg)

            return get_transform_from_preset(self.config.transform_preset)

        return build_transform(
            image_size=self.config.image_size,
            normalize=self.config.normalize,
            normalize_mean=self.config.normalize_mean,
            normalize_std=self.config.normalize_std,
            augmentation=self.config.augmentation,
            is_training=True,
        )

    def _build_val_transform(self):
        """Build the validation transform defined by the config."""
        self._require_torch()
        from li_4yp.utils import build_transform

        return build_transform(
            image_size=self.config.image_size,
            normalize=self.config.normalize,
            normalize_mean=self.config.normalize_mean,
            normalize_std=self.config.normalize_std,
            is_training=False,
        )

    def _resolve_split_indices(self, dataset) -> tuple[list[int], list[int]]:
        """Resolve train/validation indices for either fixed or random splits."""
        val_fold_idx = getattr(self.config, "val_fold_idx", None)

        if val_fold_idx is not None:
            self.logger.info(f"Using fixed split: Fold {val_fold_idx} as validation.")

            if not hasattr(dataset, "df"):
                raise AttributeError(
                    "Dataset must expose a .df attribute to use val_fold_idx."
                )

            if "fold" not in dataset.df.columns:
                raise ValueError(
                    "CSV must contain a 'fold' column when val_fold_idx is set."
                )

            val_indices = dataset.df.index[dataset.df["fold"] == val_fold_idx].tolist()
            train_indices = dataset.df.index[
                dataset.df["fold"] != val_fold_idx
            ].tolist()

            if len(val_indices) == 0:
                raise ValueError(
                    f"No samples found for fold {val_fold_idx}. Check the CSV fold values."
                )

            return train_indices, val_indices

        self.logger.info("Using random split (no 'val_fold_idx' in config).")
        torch, _, _, _, random_split = self._require_torch()
        train_size = int(self.config.train_split * len(dataset))
        val_size = len(dataset) - train_size
        train_subset, val_subset = random_split(
            range(train_size + val_size),
            [train_size, val_size],
            generator=torch.Generator().manual_seed(self.config.random_seed),
        )
        return train_subset.indices, val_subset.indices

    def _unpack_batch(self, batch):
        """Move batch tensors to the current device and track batch size."""
        if len(batch) == 3:
            images, features, labels = batch
            images = images.to(self.device)
            features = features.to(self.device)
            labels = labels.to(self.device)
            return images, features, labels, images.size(0)

        if len(batch) == 2:
            images, labels = batch
            images = images.to(self.device)
            labels = labels.to(self.device)
            return images, None, labels, images.size(0)

        raise ValueError(
            f"Expected batches of length 2 or 3, got batch of length {len(batch)}."
        )

    def _forward_batch(self, images, features):
        """Dispatch the model call for image-only or hybrid batches."""
        if features is not None:
            return self.model(images, features)
        return self.model(images)

    def _validate_loss_weights(self) -> None:
        """Validate loss weights for three-head multitask outputs."""
        if len(self.config.loss_weights) != 3:
            raise ValueError(
                "loss_weights must contain exactly three values for "
                f"base/primary/secondary tasks, got {self.config.loss_weights}"
            )

    def _compute_pytorch_loss(self, outputs, labels):
        """Resolve the correct training loss for model outputs."""
        if self.loss_fn is not None:
            return self.loss_fn(outputs, labels)

        if isinstance(outputs, dict):
            from li_4yp.training import multitask_loss

            self._validate_loss_weights()
            return multitask_loss(outputs, labels, self.config.loss_weights)

        from li_4yp.training import compute_loss

        return compute_loss(outputs, labels)

    def _build_eval_predictions_and_loss(self, outputs, labels):
        """Build evaluation predictions while preserving the current loss behavior."""
        if isinstance(outputs, dict):
            torch, nn, _, _, _ = self._require_torch()
            self._validate_loss_weights()
            criterion = nn.CrossEntropyLoss()
            loss = sum(
                criterion(outputs[key], labels[:, i]) * self.config.loss_weights[i]
                for i, key in enumerate(["base", "primary", "secondary"])
                if key in outputs
            )
            preds = torch.stack(
                [outputs[key].argmax(dim=1) for key in ["base", "primary", "secondary"]],
                dim=1,
            )
            return loss, preds

        from li_4yp.training import compute_loss

        loss = compute_loss(outputs, labels)
        preds = outputs.argmax(dim=-1)
        return loss, preds

    def setup_data(
        self, dataset_class: Optional[type] = None, **dataset_kwargs
    ) -> None:
        """Setup data. Supports both random split and fixed-fold split."""
        self.logger.info("Setting up data loaders...")

        if self.config.model_type == "pytorch":
            _, _, DataLoader, Subset, _ = self._require_torch()
            if dataset_class is None:
                raise ValueError("dataset_class must be provided for PyTorch models.")

            self.train_transform = self._build_train_transform()
            self.val_transform = self._build_val_transform()

            metadata_dataset = dataset_class(
                data_dir=self.config.data_dir,
                csv_file=self.config.csv_file,
                **dataset_kwargs,
            )

            data_info = {
                "dataset_class": dataset_class.__name__,
                "total_samples": len(metadata_dataset),
                "data_dir": self.config.data_dir,
                "csv_file": self.config.csv_file,
            }
            if hasattr(metadata_dataset, "get_info"):
                data_info.update(metadata_dataset.get_info())
            self.logger.log_data_info(data_info)

            train_indices, val_indices = self._resolve_split_indices(metadata_dataset)

            train_dataset_full = dataset_class(
                data_dir=self.config.data_dir,
                csv_file=self.config.csv_file,
                transform=self.train_transform,
                **dataset_kwargs,
            )
            train_dataset = Subset(train_dataset_full, train_indices)

            val_dataset_full = dataset_class(
                data_dir=self.config.data_dir,
                csv_file=self.config.csv_file,
                transform=self.val_transform,
                **dataset_kwargs,
            )
            val_dataset = Subset(val_dataset_full, val_indices)
            self.train_loader = DataLoader(
                train_dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=self.config.num_workers,
            )

            self.val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.num_workers,
            )

            self.logger.info(f"Train samples: {len(train_dataset)}")
            self.logger.info(f"Val samples: {len(val_dataset)}")

        elif self.config.model_type == "sklearn":
            import pandas as pd
            from sklearn.model_selection import train_test_split

            df = pd.read_csv(Path(self.config.data_dir) / self.config.csv_file)
            X = df[self.config.feature_cols].values
            y = df[self.config.label_cols].values

            # Split data
            self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
                X,
                y,
                test_size=1 - self.config.train_split,
                random_state=self.config.random_seed,
            )

            data_info = {
                "total_samples": len(X),
                "data_dir": self.config.data_dir,
                "csv_file": self.config.csv_file,
            }
            self.logger.log_data_info(data_info)

            self.logger.info(f"Train samples: {len(self.X_train)}")
            self.logger.info(f"Val samples: {len(self.X_val)}")
            self.logger.info(f"Feature shape: {self.X_train.shape}")

    def setup_model(self) -> None:
        """Setup model from registry or custom model."""
        self.logger.info("Setting up model...")

        # Get model from registry
        self.model = ModelRegistry.get(
            self.config.model_name, **self.config.model_params
        )

        # Log model info
        model_info = {
            "model_name": self.config.model_name,
            "model_type": self.config.model_type,
            "model_params": self.config.model_params,
        }

        if self.config.model_type == "pytorch":
            self._require_torch()
            # Move model to device
            self.model.to(self.device)

            # Count parameters
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            )
            model_info.update(
                {
                    "total_parameters": total_params,
                    "trainable_parameters": trainable_params,
                }
            )

        self.logger.log_model_info(model_info)

    def setup_optimizer(self) -> None:
        """Setup optimizer and scheduler for PyTorch models."""
        if self.config.model_type != "pytorch":
            return

        self._require_torch()
        torch, _, _, _, _ = self._require_torch()
        self.logger.info("Setting up optimizer...")

        if not hasattr(torch.optim, self.config.optimizer):
            raise ValueError(
                f"Unknown optimizer '{self.config.optimizer}'. "
                "Check torch.optim for valid optimizer names."
            )
        optimizer_class = getattr(torch.optim, self.config.optimizer)
        self.optimizer = optimizer_class(
            self.model.parameters(),
            lr=self.config.learning_rate,
            **self.config.optimizer_params,
        )

        if self.config.scheduler:
            if not hasattr(torch.optim.lr_scheduler, self.config.scheduler):
                raise ValueError(
                    f"Unknown scheduler '{self.config.scheduler}'. "
                    "Check torch.optim.lr_scheduler for valid scheduler names."
                )
            scheduler_class = getattr(torch.optim.lr_scheduler, self.config.scheduler)
            self.scheduler = scheduler_class(
                self.optimizer, **self.config.scheduler_params
            )
            self.logger.info(f"Using scheduler: {self.config.scheduler}")

    def setup_loss_fn(self) -> None:
        """Setup loss function for PyTorch models."""
        if self.config.model_type != "pytorch":
            return

        self._require_torch()
        self.logger.info("Setting up loss function...")

        self.loss_fn = LossRegistry.get(
            self.config.loss_name, **self.config.loss_params
        )

    def setup_evaluator(self) -> None:
        """Setup evaluator."""
        self.evaluator = ShadeEvaluator(**self.config.evaluator_params)
        self.logger.info(
            f"Evaluator configured with params: {self.config.evaluator_params}"
        )

    def train_pytorch_epoch(self) -> float:
        """Train one epoch for PyTorch model.

        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0

        for batch in self.train_loader:
            self.optimizer.zero_grad()
            images, features, labels, batch_size = self._unpack_batch(batch)
            outputs = self._forward_batch(images, features)
            loss = self._compute_pytorch_loss(outputs, labels)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * batch_size

        avg_loss = total_loss / len(self.train_loader.dataset)
        return avg_loss

    def evaluate_pytorch(
        self, data_loader: DataLoader, return_preds: bool = False
    ) -> Tuple[float, Dict[str, float]] | Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate PyTorch model.

        Args:
            data_loader: Data loader to evaluate on
            return_preds: Whether to return predictions and labels

        Returns:
            Tuple of (average loss, metrics dictionary) or (all_preds, all_labels) if return_preds is True
        """
        torch, _, _, _, _ = self._require_torch()
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in data_loader:
                images, features, labels, batch_size = self._unpack_batch(batch)
                outputs = self._forward_batch(images, features)
                loss, preds = self._build_eval_predictions_and_loss(outputs, labels)

                total_loss += loss.item() * batch_size
                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())

        avg_loss = total_loss / len(data_loader.dataset)
        all_preds = torch.cat(all_preds, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        self.evaluator.reset()
        self.evaluator.update(all_preds, all_labels)
        metrics = self.evaluator.summary()
        metrics = {k: v for k, v in metrics.items() if v is not None}

        if return_preds:
            return all_preds, all_labels

        return avg_loss, metrics

    def train_sklearn(self) -> None:
        """Train sklearn model."""
        self.logger.info("Training sklearn model...")

        start_time = time.time()
        self.model.fit(self.X_train, self.y_train)
        train_time = time.time() - start_time

        self.logger.info(f"Training completed in {train_time:.2f} seconds")

        y_train_pred = self.model.predict(self.X_train)
        self.evaluator.reset()
        self.evaluator.update(y_train_pred, self.y_train)
        train_metrics = self.evaluator.summary()
        train_metrics = {k: v for k, v in train_metrics.items() if v is not None}

        y_val_pred = self.model.predict(self.X_val)
        self.evaluator.reset()
        self.evaluator.update(y_val_pred, self.y_val)
        val_metrics = self.evaluator.summary()
        val_metrics = {k: v for k, v in val_metrics.items() if v is not None}

        self.logger.log_metrics("train", train_metrics)
        self.logger.log_metrics("val", val_metrics)

        if self.config.save_model:
            model_path = self.exp_dir / f"{self.config.model_name}.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(self.model, f)
            self.logger.info(f"Model saved to {model_path}")

        if self.config.save_predictions:
            self.logger.save_predictions(y_val_pred, self.y_val, phase="val")

    def train_pytorch(self) -> None:
        """Train PyTorch model."""
        torch, _, _, _, _ = self._require_torch()
        self.logger.info(f"Starting training for {self.config.num_epochs} epochs...")

        for epoch in range(1, self.config.num_epochs + 1):
            self.current_epoch = epoch

            train_loss = self.train_pytorch_epoch()
            val_loss, val_metrics = self.evaluate_pytorch(self.val_loader)

            if self.scheduler:
                self.scheduler.step(
                    val_loss
                    if "ReduceLROnPlateau" in str(type(self.scheduler))
                    else None
                )

            self.logger.log_epoch(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                metrics=val_metrics,
            )

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0

                if self.config.save_model:
                    model_path = self.exp_dir / f"{self.config.model_name}_best.pth"
                    torch.save(self.model.state_dict(), model_path)
                    self.logger.info(f"Best model saved (val_loss: {val_loss:.4f})")
            else:
                self.epochs_without_improvement += 1

            if self.epochs_without_improvement >= self.config.early_stopping_patience:
                self.logger.info(f"Early stopping triggered after {epoch} epochs")
                break

        if self.config.save_model:
            model_path = self.exp_dir / f"{self.config.model_name}_best.pth"
            if model_path.exists():
                self.model.load_state_dict(torch.load(model_path))
                self.logger.info("Loaded best model for final evaluation")

        if self.config.save_predictions:
            all_preds, all_labels = self.evaluate_pytorch(
                self.val_loader, return_preds=True
            )
            self.logger.save_predictions(all_preds, all_labels, phase="val")

    @staticmethod
    def print_transform(transform, logger):
        """Print transform details."""
        s = transform.__repr__()
        for ln in s.splitlines():
            if ln:
                logger.info(f"  {ln}")

    def run(
        self, dataset_class: Optional[type] = None, **dataset_kwargs
    ) -> Dict[str, Any]:
        """Run the complete experiment.

        Args:
            dataset_class: Dataset class to use
            transform: Optional transform
            **dataset_kwargs: Additional dataset arguments

        Returns:
            Dictionary with final metrics
        """
        self.setup_model()
        self.setup_evaluator()

        if self.config.model_type == "pytorch":
            self.setup_data(dataset_class, **dataset_kwargs)

            self.logger.info("Using train transform:")
            Experiment.print_transform(self.train_transform, self.logger)
            self.logger.info("Using val transform:")
            Experiment.print_transform(self.val_transform, self.logger)

            self.setup_optimizer()
            if self.config.loss_name:
                self.setup_loss_fn()
            self.train_pytorch()

            val_loss, val_metrics = self.evaluate_pytorch(self.val_loader)
            self.logger.log_metrics("final_val", val_metrics)

        elif self.config.model_type == "sklearn":
            self.setup_data(**dataset_kwargs)
            self.train_sklearn()
            val_metrics = self.evaluator.summary()

        self.logger.finalize(final_metrics=val_metrics)

        return val_metrics
