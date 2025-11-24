"""Metrics for evaluating model performance."""

import numpy as np
import torch
from typing import Optional

class ShadeEvaluator:
    def __init__(
            self,
            eos_token,
            weights,
            tol_base: int = 4,
        ):
        """Initialize the ShadeEvaluator.
        
        Args:
            eos_token (int): end-of-sequence token
            weights (list of float): weights for primary, secondary, tertiary digits in hierarchical score
            tol_base (int): default: 4; the base digit defining light vs dark shades (included as dark)
        """
        self.eos_token = eos_token
        self.weights = weights
        self.tol_base = tol_base
        self.reset()

    def reset(self):
        self.per_digit_acc = []
        self.hierarchical_scores = []
        self.exact_match = []
        self.base_prim_exact_match = []
        self.prim_sec_exact_match = []
        self.tol_base_acc = []

    @staticmethod
    def _mask_from_targets(targets: torch.Tensor, eos_token: int) -> torch.Tensor:
        """Generate a mask from the targets tensor.
        
        Args:
            targets (torch.Tensor): (batch_size, seq_len) target sequences
            eos_token (int): end-of-sequence token
            
        Returns:
            torch.Tensor: (batch_size, seq_len) boolean mask
        """
        return targets != eos_token
    
    @staticmethod
    def pretty_print(summary: dict) -> None:
        """Pretty print the summary of metrics.

        Args:
            summary (dict): summary of metrics
        """
        max_key_len  = len(max(summary, key=len))
        print("Metrics Summary:")
        print("-" * (max_key_len + 12))
        for k, v in summary.items():
            print(f"{k:<{max_key_len}} |{v:>10.4f}")
        print("-" * (max_key_len + 12))

    # -------
    # Metrics
    # -------
    def _per_digit_acc(
            self,
            preds: torch.Tensor | np.ndarray,
            labels: torch.Tensor | np.ndarray
        ) -> np.ndarray:
        """Compute per-digit accuracy.

        Args:
            preds (torch.Tensor | np.ndarray): (batch_size, 4) predicted shade codes
            labels (torch.Tensor | np.ndarray): (batch_size, 4) true shade codes

        Returns:
            accuracy (np.ndarray): (4,) array of accuracies for each digit
        """
        correct = (preds == labels)
        if isinstance(correct, torch.Tensor):
            correct = correct.cpu().numpy()
        per_digit_accuracy = correct.sum(axis=0) / correct.shape[0]
        return per_digit_accuracy
    
    def _tol_base_acc(
            self,
            preds: torch.Tensor | np.ndarray,
            labels: torch.Tensor | np.ndarray,
            return_mask: bool = False
        ) -> float | Optional[torch.Tensor | np.ndarray]:
        """Compute tolerance accuracy for base digit.

        Args:
            preds (torch.Tensor | np.ndarray): (batch_size, 4) predicted shade codes
            labels (torch.Tensor | np.ndarray): (batch_size, 4) true shade codes
            return_mask (bool): whether to return the boolean mask of correct predictions

        Returns:
            accuracy (float): tolerance accuracy for base digit
        """
        # Exact condition: base must equal
        exact_cond = preds[:, 0] == labels[:, 0]
        # Tolerance condition: if base is dark, base must be within tolerance
        tol_cond = (labels[:, 0] <= self.tol_base) & (abs(preds[:, 0] - labels[:, 0]) <= 1)  # tolerance ±1 for dark base digit

        if return_mask:
            return exact_cond | tol_cond
        
        # Prediction is correct if either exact or tolerance condition is met
        return (exact_cond | tol_cond).float().mean().item() \
            if isinstance(preds, torch.Tensor) \
            else (exact_cond | tol_cond).mean().item()

    def _base_prim_exact_match(
            self,
            preds: torch.Tensor | np.ndarray,
            labels: torch.Tensor | np.ndarray,
            return_mask: bool = False
        ) -> float | Optional[torch.Tensor | np.ndarray]:
        """Compute exact match accuracy for base and primary digits.

        Args:
            preds (torch.Tensor | np.ndarray): (batch_size, 4) predicted shade codes
            labels (torch.Tensor | np.ndarray): (batch_size, 4) true shade codes
        
        Returns:
            accuracy (float): exact match accuracy for base and primary digits
        """
        if isinstance(preds, torch.Tensor):
            mask = (preds[:, :2] == labels[:, :2]).all(dim=1)
        else:
            mask = (preds[:, :2] == labels[:, :2]).all(axis=1)

        if return_mask:
            return mask

        return mask.float().mean().item() \
            if isinstance(preds, torch.Tensor) \
            else mask.mean().item()
    
    def _prim_sec_exact_match(
            self,
            preds: torch.Tensor | np.ndarray,
            labels: torch.Tensor | np.ndarray,
            return_mask: bool = False
        ) -> float | Optional[torch.Tensor | np.ndarray]:
        """Compute exact match accuracy for primary and secondary digits.

        Args:
            preds (torch.Tensor | np.ndarray): (batch_size, 4) predicted shade codes
            labels (torch.Tensor | np.ndarray): (batch_size, 4) true shade codes
        
        Returns:
            accuracy (float): exact match accuracy for primary and secondary digits
        """
        if isinstance(preds, torch.Tensor):
            mask = (preds[:, 1:3] == labels[:, 1:3]).all(dim=1)
        else:
            mask = (preds[:, 1:3] == labels[:, 1:3]).all(axis=1)
        
        if return_mask:
            return mask

        return mask.float().mean().item() \
            if isinstance(preds, torch.Tensor) \
            else mask.mean().item()

    def _exact_match(
            self, 
            preds: torch.Tensor | np.ndarray, 
            labels: torch.Tensor | np.ndarray
        ) -> float:
        """Compute exact match accuracy.

        Args:
            preds (torch.Tensor | np.ndarray): (batch_size, 4) predicted shade codes
            labels (torch.Tensor | np.ndarray): (batch_size, 4) true shade codes

        Returns:
            accuracy (float): exact match accuracy
        """
        return (preds == labels).all(dim=1).float().mean().item() \
            if isinstance(preds, torch.Tensor) \
            else (preds == labels).all(axis=1).mean().item()

    def _hierarchical_score(
            self,
            preds: torch.Tensor | np.ndarray, 
            labels: torch.Tensor | np.ndarray
        ) -> float:
        """Compute hierarchical score.
        Formula: base_correct - w_p * primary_incorrect - w_s * secondary_incorrect - w_t * tertiary_incorrect

        Args:
            preds (torch.Tensor | np.ndarray): (batch_size, 4) predicted shade codes
            labels (torch.Tensor | np.ndarray): (batch_size, 4) true shade codes

        Returns:
            score (float): hierarchical score
        """
        scores = np.zeros(labels.shape[0])  # (batch_size,)
        
        base_correct = preds[:, 0] == labels[:, 0]
        if isinstance(base_correct, torch.Tensor):
            base_correct = base_correct.cpu().numpy()

        scores[base_correct] = 1.0
        for i, w in zip(np.arange(1, len(self.weights) + 1), self.weights):
            scores[preds[:, i] != labels[:, i]] -= w

        return np.clip(scores.mean().item(), 0, 1)
    
    # ------------------------------
    # Metrics Update and Computation
    # ------------------------------
    def update(self, preds: torch.Tensor | np.ndarray, labels: torch.Tensor | np.ndarray) -> None:
        """Update metrics with new predictions and labels.

        Args:
            preds (torch.Tensor | np.ndarray): (batch_size, 4) predicted shade codes
            labels (torch.Tensor | np.ndarray): (batch_size, 4) true shade codes
        """
        # type checks: both must be torch.Tensor or both must be np.ndarray
        is_torch_preds = isinstance(preds, torch.Tensor)
        is_torch_labels = isinstance(labels, torch.Tensor)
        is_np_preds = isinstance(preds, np.ndarray)
        is_np_labels = isinstance(labels, np.ndarray)

        if not ((is_torch_preds and is_torch_labels) or (is_np_preds and is_np_labels)):
            raise TypeError(f"preds and labels must be the same type: both Tensor or both ndarray, got: {type(preds)} and {type(labels)}")

        # shape check: dimensions must agree
        if preds.shape != labels.shape:
            raise ValueError(f"preds and labels must have the same shape, got {preds.shape} and {labels.shape}")

        self.per_digit_acc.append(self._per_digit_acc(preds, labels))
        self.tol_base_acc.append(self._tol_base_acc(preds, labels))
        self.hierarchical_scores.append(self._hierarchical_score(preds, labels))
        self.exact_match.append(self._exact_match(preds, labels))
        self.base_prim_exact_match.append(self._base_prim_exact_match(preds, labels))
        self.prim_sec_exact_match.append(self._prim_sec_exact_match(preds, labels))

    def summary(self) -> dict:
        """Compute summary of all metrics.

        Returns:
            dict: summary of metrics
        """
        per_digit_acc = np.mean(self.per_digit_acc, axis=0)
        summary = {
            "base_acc": per_digit_acc[0].item(),
            f"tol_base_acc (±1 for <={self.tol_base})": np.mean(self.tol_base_acc).item(),
            "primary_acc": per_digit_acc[1].item(),
            "secondary_acc": per_digit_acc[2].item(), 
        }
        if len(self.weights) == 3:
            summary["tertiary_acc"] = per_digit_acc[3].item()
        summary.update(
            {
                "Hierarchical Score": np.mean(self.hierarchical_scores).item(),
                "Exact Match": np.mean(self.exact_match).item(),
                "Base-Primary Exact Match": np.mean(self.base_prim_exact_match).item(),
                "Primary-Secondary Exact Match": np.mean(self.prim_sec_exact_match).item(),
            }
        )
        return summary


def tolerance_accuracy(preds: torch.Tensor, labels: torch.Tensor) -> np.ndarray:
    """Compute accuracy with tolerance of 1 for deep shades.

    Args:
        preds (torch.Tensor): (batch_size, 4) predicted shade codes
        labels (torch.Tensor): (batch_size, 4) true shade codes

    Returns:
        accuracy (np.ndarray): array of accuracies [overall, base, primary, secondary, tertiary]
    """
    correct_cnt = np.zeros(5, dtype=int)  # overall, base, prim, sec, tert
    total = preds.shape[0]
    dark_base = list(range(1, 6))  # 1-5

    def check_pred(pred, label, correct_cnt, base="dark") -> None:
        if all(pred == label):
            correct_cnt += 1
            return

        if base == "light":
            for i, (pred_digit, label_digit) in enumerate(zip(pred, label), start=1):
                correct_cnt[i] += 1 if pred_digit == label_digit else 0
        else:  # dark base
            if abs(pred[0] - label[0]) <= 1:
                correct_cnt[1] += 1  # base correct
            for i, (pred_digit, label_digit) in enumerate(zip(pred[1:], label[1:]), start=2):
                correct_cnt[i] += 1 if pred_digit == label_digit else 0
        
    for i in range(total):
        pred, label = preds[i], labels[i]
        if label[0] in dark_base:
            check_pred(pred, label, correct_cnt, base="dark")
        else:
            check_pred(pred, label, correct_cnt, base="light")

    accuracy = correct_cnt / total
    return accuracy
