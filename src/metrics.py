"""Metrics for evaluating model performance."""

import numpy as np
import torch


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