"""
Collate functions for MLCAFormer DataLoader.
"""
from typing import List, Tuple, Union

import torch
from torch import Tensor


def collate_fn(
    batch: List[Tuple[Tensor, Tensor]]
) -> Tuple[Tensor, Tensor]:
    """Collate function for MLCAFormer DataLoader.
    
    Args:
        batch: List of tuples (x, y) where:
            - x: torch.Tensor of shape (in_steps, n_vertex, 3)
            - y: torch.Tensor of shape (out_steps, n_vertex, 1)
    
    Returns:
        Tuple of:
            - x_batch: torch.Tensor of shape (batch_size, in_steps, n_vertex, 3)
            - y_batch: torch.Tensor of shape (batch_size, out_steps, n_vertex, 1)
    """
    x_list, y_list = zip(*batch)
    
    # Stack along the batch dimension
    x_batch = torch.stack(x_list, dim=0)
    y_batch = torch.stack(y_list, dim=0)
    
    return x_batch, y_batch


def collate_fn_with_missing(
    batch: List[Tuple[Tensor, Tensor, Tensor]]
) -> Tuple[Tensor, Tensor, Tensor]:
    """Collate function for MLCAFormer DataLoader with missing mask support.
    
    Args:
        batch: List of tuples (x, y, y_is_missing) where:
            - x: torch.Tensor of shape (in_steps, n_vertex, 3)
            - y: torch.Tensor of shape (out_steps, n_vertex, 1)
            - y_is_missing: torch.Tensor of shape (out_steps, n_vertex)
    
    Returns:
        Tuple of:
            - x_batch: torch.Tensor of shape (batch_size, in_steps, n_vertex, 3)
            - y_batch: torch.Tensor of shape (batch_size, out_steps, n_vertex, 1)
            - y_missing_batch: torch.Tensor of shape (batch_size, out_steps, n_vertex)
    """
    x_list, y_list, y_missing_list = zip(*batch)
    
    x_batch = torch.stack(x_list, dim=0)
    y_batch = torch.stack(y_list, dim=0)
    y_missing_batch = torch.stack(y_missing_list, dim=0)
    
    return x_batch, y_batch, y_missing_batch


def collate_fn_squeeze_y(
    batch: List[Tuple[Tensor, Tensor]]
) -> Tuple[Tensor, Tensor]:
    """Collate function with squeezed y output.
    
    This variant squeezes the last dimension of y if output_dim=1,
    which may be useful for certain loss functions.
    
    Args:
        batch: List of tuples (x, y)
    
    Returns:
        Tuple of:
            - x_batch: torch.Tensor of shape (batch_size, in_steps, n_vertex, 3)
            - y_batch: torch.Tensor of shape (batch_size, out_steps, n_vertex)
    """
    x_list, y_list = zip(*batch)
    
    x_batch = torch.stack(x_list, dim=0)
    y_batch = torch.stack(y_list, dim=0).squeeze(-1)
    
    return x_batch, y_batch
