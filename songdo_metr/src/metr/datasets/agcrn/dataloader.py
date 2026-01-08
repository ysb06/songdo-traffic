"""
Collate functions for AGCRN DataLoader.
"""
from typing import List, Tuple

import torch
from torch import Tensor


def collate_fn(
    batch: List[Tuple[Tensor, Tensor]]
) -> Tuple[Tensor, Tensor]:
    """Collate function for AGCRN DataLoader.
    
    Args:
        batch: List of tuples (x, y) where:
            - x: torch.Tensor of shape (in_steps, n_vertex, 1)
            - y: torch.Tensor of shape (out_steps, n_vertex, 1)
    
    Returns:
        Tuple of:
            - x_batch: torch.Tensor of shape (batch_size, in_steps, n_vertex, 1)
            - y_batch: torch.Tensor of shape (batch_size, out_steps, n_vertex, 1)
    """
    x_list, y_list = zip(*batch)
    
    # Stack along the batch dimension
    x_batch = torch.stack(x_list, dim=0)
    y_batch = torch.stack(y_list, dim=0)
    
    return x_batch, y_batch


def collate_fn_squeeze(
    batch: List[Tuple[Tensor, Tensor]]
) -> Tuple[Tensor, Tensor]:
    """Collate function with squeezed output dimensions.
    
    This variant squeezes the last dimension if output_dim=1,
    which may be useful for certain loss functions.
    
    Args:
        batch: List of tuples (x, y)
    
    Returns:
        Tuple of:
            - x_batch: torch.Tensor of shape (batch_size, in_steps, n_vertex)
            - y_batch: torch.Tensor of shape (batch_size, out_steps, n_vertex)
    """
    x_list, y_list = zip(*batch)
    
    x_batch = torch.stack(x_list, dim=0).squeeze(-1)
    y_batch = torch.stack(y_list, dim=0).squeeze(-1)
    
    return x_batch, y_batch
