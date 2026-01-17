from typing import List, Tuple
import torch


def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Collate function for STGCN DataLoader.
    
    Args:
        batch: List of tuples (x, y) where:
            - x: torch.Tensor of shape (1, n_vertex, n_his)
            - y: torch.Tensor of shape (n_vertex,)
    
    Returns:
        Tuple of:
            - x_batch: torch.Tensor of shape (batch_size, 1, n_vertex, n_his)
            - y_batch: torch.Tensor of shape (batch_size, n_vertex)
    """
    x_list, y_list = zip(*batch)
    
    # Stack along the batch dimension
    x_batch = torch.stack(x_list, dim=0)  # (batch_size, 1, n_vertex, n_his)
    y_batch = torch.stack(y_list, dim=0)  # (batch_size, n_vertex)
    
    return x_batch, y_batch


def collate_fn_with_missing(
    batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate function for STGCN DataLoader with missing mask support.
    
    Args:
        batch: List of tuples (x, y, y_missing) where:
            - x: torch.Tensor of shape (1, n_his, n_vertex)
            - y: torch.Tensor of shape (n_vertex,)
            - y_missing: torch.Tensor of shape (n_vertex,) - boolean mask
    
    Returns:
        Tuple of:
            - x_batch: torch.Tensor of shape (batch_size, 1, n_his, n_vertex)
            - y_batch: torch.Tensor of shape (batch_size, n_vertex)
            - y_missing_batch: torch.Tensor of shape (batch_size, n_vertex)
    """
    x_list, y_list, y_missing_list = zip(*batch)
    
    x_batch = torch.stack(x_list, dim=0)
    y_batch = torch.stack(y_list, dim=0)
    y_missing_batch = torch.stack(y_missing_list, dim=0)
    
    return x_batch, y_batch, y_missing_batch
