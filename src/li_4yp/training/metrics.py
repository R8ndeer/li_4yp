"""Metrics for evaluating model performance."""

from __future__ import annotations

import numpy as np
from typing import Optional


def _is_torch_tensor(value) -> bool:
    """Return True for torch tensors without importing torch at module import time."""
    module_name = getattr(value.__class__, "__module__", "")
    return module_name.startswith("torch")


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
    def pretty_print(summary: dict) -> None:
        """Pretty print the summary of metrics.

        Args:
            summary (dict): summary of metrics
        """
        max_key_len = len(max(summary, key=len))
        print("Metrics Summary:")
        print("-" * (max_key_len + 12))
        for k, v in summary.items():
            print(f"{k:<{max_key_len}} |{v:>10.4f}")
        print("-" * (max_key_len + 12))

    def _per_digit_acc(
        self, preds: "torch.Tensor" | np.ndarray, labels: "torch.Tensor" | np.ndarray
    ) -> np.ndarray:
        """Compute per-digit accuracy.

        Args:
            preds (torch.Tensor | np.ndarray): (batch_size, 4) predicted shade codes
            labels (torch.Tensor | np.ndarray): (batch_size, 4) true shade codes

        Returns:
            accuracy (np.ndarray): (4,) array of accuracies for each digit
        """
        correct = preds == labels
        if _is_torch_tensor(correct):
            correct = correct.cpu().numpy()
        per_digit_accuracy = correct.sum(axis=0) / correct.shape[0]
        return per_digit_accuracy

    def _tol_base_acc(
        self,
        preds: "torch.Tensor" | np.ndarray,
        labels: "torch.Tensor" | np.ndarray,
        return_mask: bool = False,
    ) -> float | Optional["torch.Tensor" | np.ndarray]:
        """Compute tolerance accuracy for base digit.

        Args:
            preds (torch.Tensor | np.ndarray): (batch_size, 4) predicted shade codes
            labels (torch.Tensor | np.ndarray): (batch_size, 4) true shade codes
            return_mask (bool): whether to return the boolean mask of correct predictions

        Returns:
            accuracy (float): tolerance accuracy for base digit
        """
        exact_cond = preds[:, 0] == labels[:, 0]
        tol_cond = (labels[:, 0] <= self.tol_base) & (
            abs(preds[:, 0] - labels[:, 0]) <= 1
        )

        if return_mask:
            return exact_cond | tol_cond

        return (
            (exact_cond | tol_cond).float().mean().item()
            if _is_torch_tensor(preds)
            else (exact_cond | tol_cond).mean().item()
        )

    def _base_prim_exact_match(
        self,
        preds: "torch.Tensor" | np.ndarray,
        labels: "torch.Tensor" | np.ndarray,
        return_mask: bool = False,
    ) -> float | Optional["torch.Tensor" | np.ndarray]:
        """Compute exact match accuracy for base and primary digits.

        Args:
            preds (torch.Tensor | np.ndarray): (batch_size, 4) predicted shade codes
            labels (torch.Tensor | np.ndarray): (batch_size, 4) true shade codes

        Returns:
            accuracy (float): exact match accuracy for base and primary digits
        """
        if _is_torch_tensor(preds):
            mask = (preds[:, :2] == labels[:, :2]).all(dim=1)
        else:
            mask = (preds[:, :2] == labels[:, :2]).all(axis=1)

        if return_mask:
            return mask

        return (
            mask.float().mean().item()
            if _is_torch_tensor(preds)
            else mask.mean().item()
        )

    def _prim_sec_exact_match(
        self,
        preds: "torch.Tensor" | np.ndarray,
        labels: "torch.Tensor" | np.ndarray,
        return_mask: bool = False,
    ) -> float | Optional["torch.Tensor" | np.ndarray]:
        """Compute exact match accuracy for primary and secondary digits.

        Args:
            preds (torch.Tensor | np.ndarray): (batch_size, 4) predicted shade codes
            labels (torch.Tensor | np.ndarray): (batch_size, 4) true shade codes

        Returns:
            accuracy (float): exact match accuracy for primary and secondary digits
        """
        if _is_torch_tensor(preds):
            mask = (preds[:, 1:3] == labels[:, 1:3]).all(dim=1)
        else:
            mask = (preds[:, 1:3] == labels[:, 1:3]).all(axis=1)

        if return_mask:
            return mask

        return (
            mask.float().mean().item()
            if _is_torch_tensor(preds)
            else mask.mean().item()
        )

    def _exact_match(
        self, preds: "torch.Tensor" | np.ndarray, labels: "torch.Tensor" | np.ndarray
    ) -> float:
        """Compute exact match accuracy.

        Args:
            preds (torch.Tensor | np.ndarray): (batch_size, 4) predicted shade codes
            labels (torch.Tensor | np.ndarray): (batch_size, 4) true shade codes

        Returns:
            accuracy (float): exact match accuracy
        """
        return (
            (preds == labels).all(dim=1).float().mean().item()
            if _is_torch_tensor(preds)
            else (preds == labels).all(axis=1).mean().item()
        )

    def _hierarchical_score(
        self, preds: "torch.Tensor" | np.ndarray, labels: "torch.Tensor" | np.ndarray
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
        if _is_torch_tensor(base_correct):
            base_correct = base_correct.cpu().numpy()

        scores[base_correct] = 1.0
        for i, w in zip(np.arange(1, len(self.weights) + 1), self.weights):
            scores[preds[:, i] != labels[:, i]] -= w

        return np.clip(scores.mean().item(), 0, 1)

    @staticmethod
    def _validate_inputs(
        preds: "torch.Tensor" | np.ndarray, labels: "torch.Tensor" | np.ndarray
    ) -> None:
        """Validate that predictions and labels are comparable."""
        is_torch_preds = _is_torch_tensor(preds)
        is_torch_labels = _is_torch_tensor(labels)
        is_np_preds = isinstance(preds, np.ndarray)
        is_np_labels = isinstance(labels, np.ndarray)

        if not ((is_torch_preds and is_torch_labels) or (is_np_preds and is_np_labels)):
            raise TypeError(
                f"preds and labels must be the same type: both Tensor or both ndarray, got: {type(preds)} and {type(labels)}"
            )

        if preds.shape != labels.shape:
            raise ValueError(
                f"preds and labels must have the same shape, got {preds.shape} and {labels.shape}"
            )

    def update(
        self, preds: "torch.Tensor" | np.ndarray, labels: "torch.Tensor" | np.ndarray
    ) -> None:
        """Update metrics with new predictions and labels.

        Args:
            preds (torch.Tensor | np.ndarray): (batch_size, 4) predicted shade codes
            labels (torch.Tensor | np.ndarray): (batch_size, 4) true shade codes
        """
        self._validate_inputs(preds, labels)
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
        if not self.per_digit_acc:
            raise RuntimeError("summary() called before any predictions were added.")

        per_digit_acc = np.mean(self.per_digit_acc, axis=0)
        summary = {
            "base_acc": per_digit_acc[0].item(),
            f"tol_base_acc (±1 for <={self.tol_base})": np.mean(
                self.tol_base_acc
            ).item(),
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
                "Primary-Secondary Exact Match": np.mean(
                    self.prim_sec_exact_match
                ).item(),
            }
        )
        return summary
