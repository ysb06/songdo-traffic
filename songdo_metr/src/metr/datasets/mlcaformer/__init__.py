"""
MLCAFormer dataset module for traffic prediction.

This module provides Dataset and DataModule classes for the MLCAFormer model,
which requires traffic data with temporal features (time-of-day, day-of-week).
"""
from .dataloader import collate_fn, collate_fn_squeeze_y
from .datamodule import MLCAFormerDataModule
from .dataset import MLCAFormerDataset

__all__ = [
    "MLCAFormerDataset",
    "MLCAFormerDataModule",
    "collate_fn",
    "collate_fn_squeeze_y",
]
