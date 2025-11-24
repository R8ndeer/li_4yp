"""Loss functions."""

from collections import Counter
import numpy as np
import torch
import torch.nn as nn

from ..utils import get_device


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


def compute_class_weights(df, columns=['Base', 'Primary', 'Secondary']):
    """
    Computes class weights for CrossEntropyLoss to handle imbalance.
    Formula: N_samples / (N_classes * Count_class)
    """
    class_weights = {}
    
    for col in columns:
        # Get all labels (handling the -1 -> 10 mapping you did in Dataset)
        labels = df[col].values
        if col != 'Base':
            labels = np.where(labels == -1, 10, labels) # Map empty to 10
        else:
            labels = labels - 1 # Map 1..12 to 0..11
            
        # Count frequencies
        count_dict = Counter(labels)
        n_samples = len(labels)
        n_classes = len(count_dict)
        
        # Calculate weights
        weights = torch.zeros(max(count_dict.keys()) + 1)
        for cls, count in count_dict.items():
            # Standard inverse frequency formula
            weights[cls] = n_samples / (n_classes * count)
            
        class_weights[col] = weights
        
    return class_weights


class HierarchicalShadeLoss(nn.Module):
    def __init__(self, class_weights_dict, task_weights=(1.0, 1.0, 1.0)):
        super().__init__()
        self.task_weights = task_weights
        self.device = get_device()
        
        # Create a specific criterion for each head with its own class weights
        self.criterions = {
            'base': nn.CrossEntropyLoss(
                weight=torch.FloatTensor(class_weights_dict['Base']).to(self.device)
            ),
            'primary': nn.CrossEntropyLoss(
                weight=torch.FloatTensor(class_weights_dict['Primary']).to(self.device)
            ),
            'secondary': nn.CrossEntropyLoss(
                weight=torch.FloatTensor(class_weights_dict['Secondary']).to(self.device)
            )
        }

    def forward(self, logits_dict, labels):
        """
        logits_dict: {'base': tensor, 'primary': tensor, 'secondary': tensor}
        labels: tensor of shape (Batch, 3) -> [Base, Prim, Sec]
        """
        # Base Loss
        base_loss = self.criterions['base'](logits_dict['base'], labels[:, 0])
        
        # Primary Loss
        prim_loss = self.criterions['primary'](logits_dict['primary'], labels[:, 1])
        
        # Secondary Loss
        sec_loss = self.criterions['secondary'](logits_dict['secondary'], labels[:, 2])
        
        # Weighted Sum
        total_loss = (base_loss * self.task_weights[0] +
                      prim_loss * self.task_weights[1] +
                      sec_loss * self.task_weights[2])
                      
        # return total_loss, (base_loss.item(), prim_loss.item(), sec_loss.item())
        return total_loss