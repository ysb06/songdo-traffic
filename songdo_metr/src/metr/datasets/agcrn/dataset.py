"""
AGCRN Dataset for traffic prediction.

AGCRN uses only traffic values without temporal features (ToD, DoW),
as the model learns spatial relationships through adaptive graph convolution.
"""
import logging
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class AGCRNDataset(Dataset):
    """PyTorch Dataset for AGCRN model.
    
    This dataset transforms traffic data into the format required by AGCRN,
    which uses only traffic values without temporal features.
    
    Args:
        data: Traffic data DataFrame with DatetimeIndex and sensor columns.
              Shape of values: (time_steps, n_vertex)
        in_steps: Number of input time steps (default: 12)
        out_steps: Number of output/prediction time steps (default: 12)
        
    Returns:
        x: Input tensor of shape (in_steps, n_vertex, 1) - traffic values only
        y: Target tensor of shape (out_steps, n_vertex, 1)
    """
    
    def __init__(
        self, 
        data: pd.DataFrame, 
        in_steps: int = 12, 
        out_steps: int = 12,
    ):
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.n_vertex = data.shape[1]
        self.sensor_ids = list(data.columns)
        
        # Store original data for scaling
        self.data_values = data.values.astype(np.float32)  # (time_steps, n_vertex)
        self.scaled_data = self.data_values.copy()
        
        # Pre-compute valid sample indices (no NaN in window)
        self.valid_indices = self._compute_valid_indices()
        
        logger.info(f"AGCRNDataset created: {len(self.valid_indices)} samples, "
                   f"{self.n_vertex} nodes, in_steps={in_steps}, out_steps={out_steps}")
    
    def _compute_valid_indices(self) -> np.ndarray:
        """Compute valid sample indices where no NaN exists in the window.
        
        Returns:
            Array of valid starting indices
        """
        total_window = self.in_steps + self.out_steps
        num_possible = len(self.data_values) - total_window + 1
        
        if num_possible <= 0:
            logger.warning("Data length is shorter than required window size")
            return np.array([], dtype=np.int64)
        
        # Check for NaN in each possible window
        valid_indices = []
        for i in range(num_possible):
            window = self.data_values[i:i + total_window, :]
            if not np.any(np.isnan(window)):
                valid_indices.append(i)
        
        return np.array(valid_indices, dtype=np.int64)
    
    def __len__(self) -> int:
        return len(self.valid_indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single sample.
        
        Args:
            idx: Sample index
            
        Returns:
            x: Input tensor of shape (in_steps, n_vertex, 1)
            y: Target tensor of shape (out_steps, n_vertex, 1)
        """
        t = self.valid_indices[idx]
        
        # Input: traffic data only (no temporal features)
        x = np.zeros((self.in_steps, self.n_vertex, 1), dtype=np.float32)
        x[:, :, 0] = self.scaled_data[t:t + self.in_steps, :]
        
        # Target: traffic values
        y = np.zeros((self.out_steps, self.n_vertex, 1), dtype=np.float32)
        y[:, :, 0] = self.scaled_data[t + self.in_steps:t + self.in_steps + self.out_steps, :]
        
        return torch.from_numpy(x), torch.from_numpy(y)
    
    def apply_scaler(self, scaler: MinMaxScaler | StandardScaler) -> None:
        """Apply a fitted scaler to the traffic data.
        
        Args:
            scaler: Fitted MinMaxScaler or StandardScaler instance
        """
        # Reshape for scaler: (time_steps * n_vertex, 1)
        flat_data = self.data_values.reshape(-1, 1)
        scaled_flat = scaler.transform(flat_data)
        self.scaled_data = scaled_flat.reshape(self.data_values.shape)
        
        logger.debug(f"Scaler applied to dataset with {self.n_vertex} nodes")
    
    def get_unscaled_item(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single sample with unscaled traffic values.
        
        Useful for evaluation when you need original values.
        
        Args:
            idx: Sample index
            
        Returns:
            x: Input tensor with unscaled traffic (in_steps, n_vertex, 1)
            y: Target tensor with unscaled traffic (out_steps, n_vertex, 1)
        """
        t = self.valid_indices[idx]
        
        x = np.zeros((self.in_steps, self.n_vertex, 1), dtype=np.float32)
        x[:, :, 0] = self.data_values[t:t + self.in_steps, :]  # Unscaled
        
        y = np.zeros((self.out_steps, self.n_vertex, 1), dtype=np.float32)
        y[:, :, 0] = self.data_values[t + self.in_steps:t + self.in_steps + self.out_steps, :]
        
        return torch.from_numpy(x), torch.from_numpy(y)
