"""Loss functions."""

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