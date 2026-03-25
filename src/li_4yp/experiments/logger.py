"""Experiment logger for tracking training progress and metrics."""

from pathlib import Path
from typing import Any, Dict, Optional
import json
import pandas as pd
from datetime import datetime


class ExperimentLogger:
    """Logger for experiment tracking with file and console output."""

    def __init__(
        self, log_dir: str | Path, experiment_name: str, console_output: bool = True
    ):
        """Initialize the experiment logger.

        Args:
            log_dir: Directory to save logs
            experiment_name: Name of the experiment
            console_output: Whether to print to console
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name
        self.console_output = console_output

        # Create log files
        self.log_file = self.log_dir / "experiment.log"
        self.metrics_file = self.log_dir / "metrics.csv"
        self.history_file = self.log_dir / "history.csv"

        # Initialize metrics storage
        self.metrics_history = []
        self.epoch_history = []

        # Write initial log
        self._log(f"Experiment: {experiment_name}")
        self._log(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self._log("-" * 80)

    def _log(self, message: str, level: str = "INFO") -> None:
        """Internal logging method.

        Args:
            message: Message to log
            level: Log level (INFO, WARNING, ERROR)
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] [{level}] {message}"

        # Write to file
        with open(self.log_file, "a") as f:
            f.write(log_message + "\n")

        # Print to console
        if self.console_output:
            print(log_message)

    def info(self, message: str) -> None:
        """Log info message."""
        self._log(message, level="INFO")

    def warning(self, message: str) -> None:
        """Log warning message."""
        self._log(message, level="WARNING")

    def error(self, message: str) -> None:
        """Log error message."""
        self._log(message, level="ERROR")

    def _write_json(self, path: Path, payload: Dict[str, Any]) -> None:
        """Write JSON payloads used by experiment artifacts."""
        with open(path, "w") as f:
            json.dump(payload, f, indent=2, default=str)

    def _log_key_value_block(self, title: str, payload: Dict[str, Any]) -> None:
        """Log a section of key-value pairs in a consistent format."""
        self.info(title)
        for key, value in payload.items():
            self.info(f"  {key}: {value}")
        self.info("-" * 80)

    def log_config(self, config: Dict[str, Any]) -> None:
        """Log experiment configuration.

        Args:
            config: Configuration dictionary
        """
        self._log_key_value_block("Experiment Configuration:", config)
        self._write_json(self.log_dir / "config.json", config)

    def log_epoch(
        self,
        epoch: int,
        train_loss: Optional[float] = None,
        val_loss: Optional[float] = None,
        metrics: Optional[Dict[str, float]] = None,
        additional_info: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log epoch training information.

        Args:
            epoch: Current epoch number
            train_loss: Training loss
            val_loss: Validation loss
            metrics: Dictionary of evaluation metrics
            additional_info: Additional information to log
        """
        # Prepare epoch data
        epoch_data = {"epoch": epoch}

        if train_loss is not None:
            epoch_data["train_loss"] = train_loss
        if val_loss is not None:
            epoch_data["val_loss"] = val_loss
        if metrics:
            for key, value in metrics.items():
                epoch_data[f"val_{key}"] = value
        if additional_info:
            epoch_data.update(additional_info)

        # Store in history
        self.epoch_history.append(epoch_data)

        # Log to console/file
        log_msg = f"Epoch {epoch:3d}"
        if train_loss is not None:
            log_msg += f" | Train Loss: {train_loss:.4f}"
        if val_loss is not None:
            log_msg += f" | Val Loss: {val_loss:.4f}"
        if metrics:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    log_msg += f" | {key}: {value:.4f}"

        self.info(log_msg)

        self._save_history()

    def log_metrics(
        self, phase: str, metrics: Dict[str, float], step: Optional[int] = None
    ) -> None:
        """Log evaluation metrics.

        Args:
            phase: Phase name (train, val, test)
            metrics: Dictionary of metrics
            step: Optional step/epoch number
        """
        metric_data = {"phase": phase, "timestamp": datetime.now().isoformat()}

        if step is not None:
            metric_data["step"] = step

        metric_data.update(metrics)
        self.metrics_history.append(metric_data)

        self.info("")
        self.info(f"{phase.upper()} Metrics:")
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.info(f"  {key}: {value:.4f}")
            else:
                self.info(f"  {key}: {value}")
        self.info("-" * 80)

        self._save_metrics()

    def _save_history(self) -> None:
        """Save training history to CSV."""
        if self.epoch_history:
            df = pd.DataFrame(self.epoch_history)
            df.to_csv(self.history_file, index=False)

    def _save_metrics(self) -> None:
        """Save metrics to CSV."""
        if self.metrics_history:
            df = pd.DataFrame(self.metrics_history)
            df.to_csv(self.metrics_file, index=False)

    def log_model_info(self, model_info: Dict[str, Any]) -> None:
        """Log model architecture and parameters.

        Args:
            model_info: Dictionary containing model information
        """
        self._log_key_value_block("Model Information:", model_info)
        self._write_json(self.log_dir / "model_info.json", model_info)

    def log_data_info(self, data_info: Dict[str, Any]) -> None:
        """Log dataset information.

        Args:
            data_info: Dictionary containing dataset information
        """
        self._log_key_value_block("Dataset Information:", data_info)
        self._write_json(self.log_dir / "data_info.json", data_info)

    def save_predictions(
        self, predictions: Any, targets: Any, phase: str = "test"
    ) -> None:
        """Save predictions and targets.

        Args:
            predictions: Model predictions
            targets: Ground truth targets
            phase: Phase name (train, val, test)
        """

        if hasattr(predictions, "cpu"):
            predictions = predictions.cpu().numpy()
        if hasattr(targets, "cpu"):
            targets = targets.cpu().numpy()

        if len(predictions) != len(targets):
            raise ValueError(
                "predictions and targets must have the same number of rows before "
                f"saving, got {len(predictions)} and {len(targets)}."
            )

        pred_file = self.log_dir / f"{phase}_predictions.csv"
        df = pd.DataFrame(
            {
                "prediction": (
                    predictions.tolist() if predictions.ndim > 1 else predictions
                ),
                "target": targets.tolist() if targets.ndim > 1 else targets,
            }
        )
        df.to_csv(pred_file, index=False)

        self.info(f"Saved {phase} predictions to {pred_file}")

    def finalize(self, final_metrics: Optional[Dict[str, float]] = None) -> None:
        """Finalize the experiment and save summary.

        Args:
            final_metrics: Final test metrics
        """
        self.info("=" * 80)
        self.info("Experiment Completed")
        self.info(f"Ended at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        if final_metrics:
            self.info("")
            self.info("Final Test Metrics:")
            for key, value in final_metrics.items():
                if isinstance(value, (int, float)):
                    self.info(f"  {key}: {value:.4f}")
                else:
                    self.info(f"  {key}: {value}")

        self.info("=" * 80)

        summary = {
            "experiment_name": self.experiment_name,
            "completed_at": datetime.now().isoformat(),
            "total_epochs": len(self.epoch_history),
            "final_metrics": final_metrics or {},
        }
        self._write_json(self.log_dir / "summary.json", summary)

    def get_best_epoch(
        self, metric: str = "val_loss", mode: str = "min"
    ) -> Dict[str, Any]:
        """Get the best epoch based on a metric.

        Args:
            metric: Metric name to optimize
            mode: "min" or "max"

        Returns:
            Dictionary containing best epoch information
        """
        if not self.epoch_history:
            return {}

        df = pd.DataFrame(self.epoch_history)
        if metric not in df.columns:
            self.warning(f"Metric '{metric}' not found in history")
            return {}

        if mode == "min":
            best_idx = df[metric].idxmin()
        else:
            best_idx = df[metric].idxmax()

        best_epoch = df.loc[best_idx].to_dict()
        return best_epoch
