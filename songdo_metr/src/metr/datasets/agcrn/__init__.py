"""
AGCRN dataset module for traffic prediction.

This module provides Dataset and DataModule classes for the AGCRN model,
which uses adaptive graph convolution to learn spatial relationships
from traffic data without requiring external graph structures.

AGCRN (Adaptive Graph Convolutional Recurrent Network) learns:
- Node embeddings that capture spatial relationships
- Adaptive adjacency matrices through node embedding similarity
- Temporal patterns through GRU-based recurrent units
"""
from .dataloader import collate_fn, collate_fn_squeeze
from .datamodule import AGCRNDataModule
from .dataset import AGCRNDataset

__all__ = [
    "AGCRNDataset",
    "AGCRNDataModule",
    "collate_fn",
    "collate_fn_squeeze",
]
