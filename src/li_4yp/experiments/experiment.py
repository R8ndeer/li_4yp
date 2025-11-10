"""Main Experiment class for running and tracking experiments."""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np
import pickle
import time

from .config import ExperimentConfig
from .logger import ExperimentLogger
from .registry import ModelRegistry
from li_4yp.training import ShadeEvaluator
from li_4yp.utils import seed_all


class Experiment:
    """Main experiment class for training and evaluation."""
    
    def __init__(self, config: ExperimentConfig):
        """Initialize experiment.
        
        Args:
            config: Experiment configuration
        """
        self.config = config
        
        # Set random seed
        seed_all(config.random_seed)
        
        # Create experiment directory
        self.exp_dir = config.get_experiment_dir()
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize logger
        self.logger = ExperimentLogger(
            log_dir=self.exp_dir,
            experiment_name=config.name
        )
        
        # Log configuration
        self.logger.log_config(config.to_dict())
        
        # Initialize components
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.evaluator = None
        self.device = self._get_device()
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        
        self.logger.info(f"Experiment directory: {self.exp_dir}")
        self.logger.info(f"Device: {self.device}")
    
    def _get_device(self) -> torch.device:
        """Get compute device."""
        if self.config.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(self.config.device)
        return device
    
    def setup_data(
        self,
        dataset_class: type,
        transform: Optional[Any] = None,
        **dataset_kwargs
    ) -> None:
        """Setup data loaders.
        
        Args:
            dataset_class: Dataset class to use
            transform: Optional transform to apply
            **dataset_kwargs: Additional arguments for dataset
        """
        self.logger.info("Setting up data loaders...")
        
        # Create dataset
        dataset = dataset_class(
            data_dir=self.config.data_dir,
            csv_file=self.config.csv_file,
            transform=transform,
            **dataset_kwargs
        )
        
        # Log dataset info
        data_info = {
            "dataset_class": dataset_class.__name__,
            "total_samples": len(dataset),
            "data_dir": self.config.data_dir,
            "csv_file": self.config.csv_file
        }
        
        if hasattr(dataset, 'get_info'):
            data_info.update(dataset.get_info())
        
        self.logger.log_data_info(data_info)
        
        # For PyTorch models, create data loaders
        if self.config.model_type == "pytorch":
            # Split dataset
            train_size = int(self.config.train_split * len(dataset))
            val_size = len(dataset) - train_size
            train_dataset, val_dataset = random_split(
                dataset,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(self.config.random_seed)
            )
            
            # Create data loaders
            self.train_loader = DataLoader(
                train_dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=self.config.num_workers
            )
            
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.num_workers
            )
            
            self.logger.info(f"Train samples: {train_size}")
            self.logger.info(f"Val samples: {val_size}")
        
        # For sklearn models, prepare data
        elif self.config.model_type == "sklearn":
            # Extract features and labels
            from sklearn.model_selection import train_test_split
            
            X = []
            y = []
            for i in range(len(dataset)):
                features, labels = dataset[i]
                # Handle different data formats
                if isinstance(features, torch.Tensor):
                    features = features.numpy()
                if isinstance(labels, torch.Tensor):
                    labels = labels.numpy()
                X.append(features.flatten())  # Flatten image if needed
                y.append(labels)
            
            X = np.array(X)
            y = np.array(y)
            
            # Split data
            self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
                X, y,
                test_size=1 - self.config.train_split,
                random_state=self.config.random_seed
            )
            
            self.logger.info(f"Train samples: {len(self.X_train)}")
            self.logger.info(f"Val samples: {len(self.X_val)}")
            self.logger.info(f"Feature shape: {self.X_train.shape}")
    
    def setup_model(self) -> None:
        """Setup model from registry or custom model."""
        self.logger.info("Setting up model...")
        
        # Get model from registry
        self.model = ModelRegistry.get(
            self.config.model_name,
            **self.config.model_params
        )
        
        # Log model info
        model_info = {
            "model_name": self.config.model_name,
            "model_type": self.config.model_type,
            "model_params": self.config.model_params
        }
        
        if self.config.model_type == "pytorch":
            # Move model to device
            self.model.to(self.device)
            
            # Count parameters
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            model_info.update({
                "total_parameters": total_params,
                "trainable_parameters": trainable_params
            })
        
        self.logger.log_model_info(model_info)
    
    def setup_optimizer(self) -> None:
        """Setup optimizer and scheduler for PyTorch models."""
        if self.config.model_type != "pytorch":
            return
        
        self.logger.info("Setting up optimizer...")
        
        # Get optimizer class
        optimizer_class = getattr(torch.optim, self.config.optimizer)
        self.optimizer = optimizer_class(
            self.model.parameters(),
            lr=self.config.learning_rate,
            **self.config.optimizer_params
        )
        
        # Setup scheduler if specified
        if self.config.scheduler:
            scheduler_class = getattr(torch.optim.lr_scheduler, self.config.scheduler)
            self.scheduler = scheduler_class(
                self.optimizer,
                **self.config.scheduler_params
            )
            self.logger.info(f"Using scheduler: {self.config.scheduler}")
    
    def setup_evaluator(self) -> None:
        """Setup evaluator."""
        self.evaluator = ShadeEvaluator(**self.config.evaluator_params)
        self.logger.info(f"Evaluator configured with params: {self.config.evaluator_params}")
    
    def train_pytorch_epoch(self) -> float:
        """Train one epoch for PyTorch model.
        
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        
        for images, labels in self.train_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(images)
            
            # Compute loss (customize based on model output format)
            if isinstance(outputs, dict):
                from li_4yp.training import multitask_loss
                loss = multitask_loss(outputs, labels, self.config.loss_weights)
            else:
                from li_4yp.training import compute_loss
                loss = compute_loss(outputs, labels)  # standard CE loss
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item() * images.size(0)
        
        avg_loss = total_loss / len(self.train_loader.dataset)
        return avg_loss
    
    def evaluate_pytorch(self, data_loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """Evaluate PyTorch model.
        
        Args:
            data_loader: Data loader to evaluate on
            
        Returns:
            Tuple of (average loss, metrics dictionary)
        """
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in data_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                
                # Compute loss
                if isinstance(outputs, dict):
                    criterion = nn.CrossEntropyLoss()
                    loss = sum(
                        criterion(outputs[key], labels[:, i]) * self.config.loss_weights[i]
                        for i, key in enumerate(['base', 'primary', 'secondary'])
                        if key in outputs
                    )
                    # Convert dict outputs to tensor
                    preds = torch.stack([
                        outputs[key].argmax(dim=1)
                        for key in ['base', 'primary', 'secondary']
                    ], dim=1)
                else:
                    from li_4yp.training import compute_loss
                    loss = compute_loss(outputs, labels)
                    preds = outputs.argmax(dim=-1)
                
                total_loss += loss.item() * images.size(0)
                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())
        
        avg_loss = total_loss / len(data_loader.dataset)
        all_preds = torch.cat(all_preds, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        # Compute metrics
        self.evaluator.reset()
        self.evaluator.update(all_preds, all_labels)
        metrics = self.evaluator.summary()
        
        # Remove None values
        metrics = {k: v for k, v in metrics.items() if v is not None}
        
        return avg_loss, metrics
    
    def train_sklearn(self) -> None:
        """Train sklearn model."""
        self.logger.info("Training sklearn model...")
        
        start_time = time.time()
        self.model.fit(self.X_train, self.y_train)
        train_time = time.time() - start_time
        
        self.logger.info(f"Training completed in {train_time:.2f} seconds")
        
        # Evaluate on training set
        y_train_pred = self.model.predict(self.X_train)
        self.evaluator.reset()
        self.evaluator.update(y_train_pred, self.y_train)
        train_metrics = self.evaluator.summary()
        train_metrics = {k: v for k, v in train_metrics.items() if v is not None}
        
        # Evaluate on validation set
        y_val_pred = self.model.predict(self.X_val)
        self.evaluator.reset()
        self.evaluator.update(y_val_pred, self.y_val)
        val_metrics = self.evaluator.summary()
        val_metrics = {k: v for k, v in val_metrics.items() if v is not None}
        
        # Log metrics
        self.logger.log_metrics("train", train_metrics)
        self.logger.log_metrics("val", val_metrics)
        
        # Save model
        if self.config.save_model:
            model_path = self.exp_dir / f"{self.config.model_name}.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(self.model, f)
            self.logger.info(f"Model saved to {model_path}")
        
        # Save predictions
        if self.config.save_predictions:
            self.logger.save_predictions(y_val_pred, self.y_val, phase="val")
    
    def train_pytorch(self) -> None:
        """Train PyTorch model."""
        self.logger.info(f"Starting training for {self.config.num_epochs} epochs...")
        
        for epoch in range(1, self.config.num_epochs + 1):
            self.current_epoch = epoch
            
            # Train one epoch
            train_loss = self.train_pytorch_epoch()
            
            # Evaluate on validation set
            val_loss, val_metrics = self.evaluate_pytorch(self.val_loader)
            
            # Update scheduler
            if self.scheduler:
                self.scheduler.step(val_loss if 'ReduceLROnPlateau' in str(type(self.scheduler)) else None)
            
            # Log epoch
            self.logger.log_epoch(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                metrics=val_metrics
            )
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0
                
                if self.config.save_model:
                    model_path = self.exp_dir / f"{self.config.model_name}_best.pth"
                    torch.save(self.model.state_dict(), model_path)
                    self.logger.info(f"Best model saved (val_loss: {val_loss:.4f})")
            else:
                self.epochs_without_improvement += 1
            
            # Early stopping
            if self.epochs_without_improvement >= self.config.early_stopping_patience:
                self.logger.info(f"Early stopping triggered after {epoch} epochs")
                break
        
        # Load best model for final evaluation
        if self.config.save_model:
            model_path = self.exp_dir / f"{self.config.model_name}_best.pth"
            if model_path.exists():
                self.model.load_state_dict(torch.load(model_path))
                self.logger.info("Loaded best model for final evaluation")
    
    def run(
        self,
        dataset_class: type,
        transform: Optional[Any] = None,
        **dataset_kwargs
    ) -> Dict[str, Any]:
        """Run the complete experiment.
        
        Args:
            dataset_class: Dataset class to use
            transform: Optional transform
            **dataset_kwargs: Additional dataset arguments
            
        Returns:
            Dictionary with final metrics
        """
        # Setup
        self.setup_data(dataset_class, transform, **dataset_kwargs)
        self.setup_model()
        self.setup_evaluator()
        
        # Train
        if self.config.model_type == "pytorch":
            self.setup_optimizer()
            self.train_pytorch()
            
            # Final evaluation
            val_loss, val_metrics = self.evaluate_pytorch(self.val_loader)
            self.logger.log_metrics("final_val", val_metrics)
            
        elif self.config.model_type == "sklearn":
            self.train_sklearn()
            val_metrics = self.evaluator.summary()
        
        # Finalize
        self.logger.finalize(final_metrics=val_metrics)
        
        return val_metrics
