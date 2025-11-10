"""Loss functions."""

import torch
import torch.nn as nn


def compute_loss(logits, labels):
    """Compute the CE loss for RNN outputs.

    Args:
        logits: (batch_size, seq_len, vocab_size)
        labels: (batch_size, seq_len)
    """
    criterion = nn.CrossEntropyLoss()
    batched_labels = labels.view(-1)  # (batch_size * seq_len)
    batched_logits = logits.view(-1, logits.shape[-1])  # (batch_size * seq_len, vocab_size)
    loss = criterion(batched_logits, batched_labels)
    return loss


def multitask_loss(logits_dict: dict, labels: torch.Tensor, weights: tuple) -> float:
    """Compute the multitask loss for CNN-like models.

    Args:
        logits_dict (dict): Dictionary of task logits
        labels (torch.Tensor): (batch_size, seq_len) true labels

    Returns:
        torch.Tensor: Computed loss
    """
    criterion = nn.CrossEntropyLoss()
    assert len(logits_dict) == labels.shape[1] == len(weights)
    total_loss = 0.0
    for i, (_, logits) in enumerate(logits_dict.items()):
        task_loss = criterion(logits, labels[:, i])
        total_loss += task_loss * weights[i]
    return total_loss