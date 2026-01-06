from typing import List, Tuple

import torch


def collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor]]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Collate function for DCRNN DataLoader.

    Transforms batch-first data to time-first format and flattens node features
    as required by DCRNNModel.forward().

    Args:
        batch: List of tuples (x, y) where:
            - x: torch.Tensor of shape (seq_len, num_nodes, input_dim)
            - y: torch.Tensor of shape (horizon, num_nodes, output_dim)

    Returns:
        Tuple of:
            - x_batch: torch.Tensor of shape (seq_len, batch_size, num_nodes * input_dim)
            - y_batch: torch.Tensor of shape (horizon, batch_size, num_nodes * output_dim)
    """
    x_list, y_list = zip(*batch)

    # Stack along batch dimension
    # x_stacked: (batch_size, seq_len, num_nodes, input_dim)
    # y_stacked: (batch_size, horizon, num_nodes, output_dim)
    x_stacked = torch.stack(x_list, dim=0)
    y_stacked = torch.stack(y_list, dim=0)

    batch_size = x_stacked.size(0)
    seq_len = x_stacked.size(1)
    num_nodes = x_stacked.size(2)
    input_dim = x_stacked.size(3)
    horizon = y_stacked.size(1)
    output_dim = y_stacked.size(3)

    # Reshape and transpose to time-first format
    # From: (batch_size, seq_len, num_nodes, input_dim)
    # To: (seq_len, batch_size, num_nodes * input_dim)
    x_batch = x_stacked.permute(1, 0, 2, 3).reshape(seq_len, batch_size, num_nodes * input_dim)

    # From: (batch_size, horizon, num_nodes, output_dim)
    # To: (horizon, batch_size, num_nodes * output_dim)
    y_batch = y_stacked.permute(1, 0, 2, 3).reshape(horizon, batch_size, num_nodes * output_dim)

    return x_batch, y_batch


def collate_fn_with_metadata(
    batch: List[Tuple[torch.Tensor, torch.Tensor]]
) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    """Collate function that also returns metadata for debugging/analysis.

    Args:
        batch: List of tuples (x, y)

    Returns:
        Tuple of:
            - x_batch: torch.Tensor of shape (seq_len, batch_size, num_nodes * input_dim)
            - y_batch: torch.Tensor of shape (horizon, batch_size, num_nodes * output_dim)
            - metadata: dict with shape information
    """
    x_batch, y_batch = collate_fn(batch)

    # Extract original shapes from first sample
    x_sample, y_sample = batch[0]

    metadata = {
        "batch_size": len(batch),
        "seq_len": x_sample.size(0),
        "horizon": y_sample.size(0),
        "num_nodes": x_sample.size(1),
        "input_dim": x_sample.size(2),
        "output_dim": y_sample.size(2),
    }

    return x_batch, y_batch, metadata
