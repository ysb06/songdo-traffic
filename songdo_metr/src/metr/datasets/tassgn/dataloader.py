"""
Collate functions for TASSGN DataLoaders.

This module provides phase-specific collate functions:
- encoder_collate_fn: Phase 1 - STIDEncoder pre-training
- predictor_collate_fn: Phase 3 - Predictor training
- tassgn_collate_fn: Phase 5 - Final TASSGN model training
"""
from typing import List, Tuple

import torch
from torch import Tensor


def encoder_collate_fn(batch: List[Tensor]) -> Tuple[Tensor, Tensor]:
    """Collate function for Phase 1: STIDEncoder pre-training.
    
    STIDEncoder's pretrain() method receives the same tensor as both input and target.
    The input gets masked internally, and the model learns to reconstruct the original.
    
    Args:
        batch: List of tensors y where:
            - y: torch.Tensor of shape (out_steps, n_vertex, 3)
    
    Returns:
        Tuple of:
            - y_input: torch.Tensor of shape (batch_size, out_steps, n_vertex, 3)
                      This will be masked inside the encoder
            - y_target: torch.Tensor of shape (batch_size, out_steps, n_vertex, 1)
                       Target for reconstruction loss (only traffic values)
    """
    y_batch = torch.stack(batch, dim=0)  # (B, out_steps, n_vertex, 3)
    
    # For reconstruction, target is only the traffic values
    y_target = y_batch[..., 0:1]  # (B, out_steps, n_vertex, 1)
    
    return y_batch, y_target


def predictor_collate_fn(batch: List[Tuple[Tensor, Tensor]]) -> Tuple[Tensor, Tensor]:
    """Collate function for Phase 3: Predictor training.
    
    Predictor learns to classify future patterns from history series.
    
    Args:
        batch: List of tuples (x, labels) where:
            - x: torch.Tensor of shape (in_steps, n_vertex, 3)
            - labels: torch.Tensor of shape (n_vertex, 1)
    
    Returns:
        Tuple of:
            - x_batch: torch.Tensor of shape (batch_size, in_steps, n_vertex, 3)
            - labels_batch: torch.Tensor of shape (batch_size, n_vertex, 1)
    """
    x_list, labels_list = zip(*batch)
    
    x_batch = torch.stack(x_list, dim=0)
    labels_batch = torch.stack(labels_list, dim=0)
    
    return x_batch, labels_batch


def tassgn_collate_fn(batch: List[Tuple[Tensor, Tensor, Tensor]]) -> Tuple[Tensor, Tensor, Tensor]:
    """Collate function for Phase 5: Final TASSGN model training.
    
    TASSGN model takes history series, self-sampled patterns, and predicts future.
    
    Args:
        batch: List of tuples (x, sample_data, y) where:
            - x: torch.Tensor of shape (in_steps, n_vertex, 3)
            - sample_data: torch.Tensor of shape (num_samples, out_steps, n_vertex, 3)
            - y: torch.Tensor of shape (out_steps, n_vertex, 1)
    
    Returns:
        Tuple of:
            - x_batch: torch.Tensor of shape (batch_size, in_steps, n_vertex, 3)
            - sample_batch: torch.Tensor of shape (batch_size, num_samples, out_steps, n_vertex, 3)
            - y_batch: torch.Tensor of shape (batch_size, out_steps, n_vertex, 1)
    """
    x_list, sample_list, y_list = zip(*batch)
    
    x_batch = torch.stack(x_list, dim=0)
    sample_batch = torch.stack(sample_list, dim=0)
    y_batch = torch.stack(y_list, dim=0)
    
    return x_batch, sample_batch, y_batch


def window_collate_fn(batch: List[Tensor]) -> Tensor:
    """Collate function for WindowDataset (utility for Phase 4).
    
    Args:
        batch: List of tensors where:
            - window: torch.Tensor of shape (out_steps, n_vertex, 3)
    
    Returns:
        window_batch: torch.Tensor of shape (batch_size, out_steps, n_vertex, 3)
    """
    return torch.stack(batch, dim=0)
