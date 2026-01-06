"""
TASSGN Dataset classes for traffic prediction.

This module provides Dataset classes for each phase of TASSGN training:
- TASSGNBaseDataset: Base class with common functionality
- EncoderDataset: Phase 1 - STIDEncoder pre-training
- PredictorDataset: Phase 3 - Predictor training
- TASSGNDataset: Phase 5 - Final TASSGN model training
"""
import logging
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class TASSGNBaseDataset(Dataset):
    """Base PyTorch Dataset for TASSGN model.
    
    This base class provides common functionality for all TASSGN phases:
    - Raw data loading from DataFrame
    - Sliding window generation
    - Time-of-Day (ToD) and Day-of-Week (DoW) extraction (TASSGN format: integers)
    - Valid sample index computation
    
    Args:
        data: Traffic data DataFrame with DatetimeIndex and sensor columns.
              Shape of values: (time_steps, n_vertex)
        in_steps: Number of input time steps (default: 12)
        out_steps: Number of output/prediction time steps (default: 12)
        steps_per_day: Number of time steps per day (default: 288 for 5-min intervals)
        
    Attributes:
        data_values: Original traffic data array of shape (time_steps, n_vertex)
        scaled_data: Scaled traffic data array
        tod: Time-of-Day features as integers [0, steps_per_day-1]
        dow: Day-of-Week features as integers [0, 6]
        valid_indices: Array of valid starting indices (no NaN in window)
    """
    
    def __init__(
        self, 
        data: pd.DataFrame, 
        in_steps: int = 12, 
        out_steps: int = 12,
        steps_per_day: int = 288,
    ):
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.steps_per_day = steps_per_day
        self.n_vertex = data.shape[1]
        self.sensor_ids = list(data.columns)
        
        # Store original data for scaling
        self.data_values = data.values.astype(np.float32)  # (time_steps, n_vertex)
        self.scaled_data = self.data_values.copy()
        
        # Extract temporal features from DatetimeIndex (TASSGN format: integers)
        self.tod = self._extract_tod(data.index)  # (time_steps,) - integers [0, steps_per_day-1]
        self.dow = self._extract_dow(data.index)  # (time_steps,) - integers [0, 6]
        
        # Pre-compute valid sample indices (no NaN in window)
        self.valid_indices = self._compute_valid_indices()
        
        logger.info(f"TASSGNBaseDataset created: {len(self.valid_indices)} samples, "
                   f"{self.n_vertex} nodes, in_steps={in_steps}, out_steps={out_steps}")
    
    def _extract_tod(self, index: pd.DatetimeIndex) -> np.ndarray:
        """Extract Time-of-Day feature as integer indices.
        
        TASSGN uses integer ToD for embedding lookup, not continuous values.
        
        Args:
            index: DatetimeIndex of the data
            
        Returns:
            Array of shape (time_steps,) with values in [0, steps_per_day-1]
        """
        # Calculate step index within day based on steps_per_day
        minutes_per_step = 24 * 60 / self.steps_per_day
        minutes = index.hour * 60 + index.minute
        step_in_day = (minutes / minutes_per_step).astype(np.int32)
        return step_in_day.values
    
    def _extract_dow(self, index: pd.DatetimeIndex) -> np.ndarray:
        """Extract Day-of-Week feature as integers [0, 6].
        
        Args:
            index: DatetimeIndex of the data
            
        Returns:
            Array of shape (time_steps,) with values in [0, 6]
        """
        return index.dayofweek.values.astype(np.int32)
    
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
    
    def _get_x_features(self, t: int) -> np.ndarray:
        """Get input features (x) for a given starting time index.
        
        Args:
            t: Starting time index
            
        Returns:
            Array of shape (in_steps, n_vertex, 3) with [traffic, tod, dow]
        """
        x = np.zeros((self.in_steps, self.n_vertex, 3), dtype=np.float32)
        
        # Traffic data (scaled)
        x[:, :, 0] = self.scaled_data[t:t + self.in_steps, :]
        
        # Time-of-Day: broadcast to all nodes
        x[:, :, 1] = self.tod[t:t + self.in_steps, np.newaxis]
        
        # Day-of-Week: broadcast to all nodes
        x[:, :, 2] = self.dow[t:t + self.in_steps, np.newaxis]
        
        return x
    
    def _get_y_features(self, t: int, include_temporal: bool = True) -> np.ndarray:
        """Get target features (y) for a given starting time index.
        
        Args:
            t: Starting time index (beginning of input window)
            include_temporal: Whether to include ToD/DoW features (default: True)
            
        Returns:
            Array of shape (out_steps, n_vertex, 3) if include_temporal
            Array of shape (out_steps, n_vertex, 1) if not include_temporal
        """
        y_start = t + self.in_steps
        
        if include_temporal:
            y = np.zeros((self.out_steps, self.n_vertex, 3), dtype=np.float32)
            y[:, :, 0] = self.scaled_data[y_start:y_start + self.out_steps, :]
            y[:, :, 1] = self.tod[y_start:y_start + self.out_steps, np.newaxis]
            y[:, :, 2] = self.dow[y_start:y_start + self.out_steps, np.newaxis]
        else:
            y = np.zeros((self.out_steps, self.n_vertex, 1), dtype=np.float32)
            y[:, :, 0] = self.scaled_data[y_start:y_start + self.out_steps, :]
        
        return y
    
    def apply_scaler(self, scaler: Union[MinMaxScaler, StandardScaler]) -> None:
        """Apply a fitted scaler to the traffic data.
        
        Note: Only traffic values are scaled. ToD and DoW remain unchanged.
        
        Args:
            scaler: Fitted scaler instance (MinMaxScaler or StandardScaler)
        """
        # Reshape for scaler: (time_steps * n_vertex, 1)
        flat_data = self.data_values.reshape(-1, 1)
        scaled_flat = scaler.transform(flat_data)
        self.scaled_data = scaled_flat.reshape(self.data_values.shape)
        
        logger.debug(f"Scaler applied to dataset with {self.n_vertex} nodes")
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single sample. Override in subclasses for specific formats."""
        raise NotImplementedError("Subclasses must implement __getitem__")


class EncoderDataset(TASSGNBaseDataset):
    """Dataset for Phase 1: STIDEncoder pre-training.
    
    STIDEncoder learns to reconstruct masked future time series (y).
    This dataset provides future series data for self-supervised learning.
    
    Note: The encoder receives y_data as input (which gets masked internally),
    and learns to reconstruct the original y_data.
    
    Args:
        data: Traffic data DataFrame
        in_steps: Not used for encoder (kept for consistency)
        out_steps: Length of future series (default: 12)
        steps_per_day: Number of time steps per day (default: 288)
        
    Returns:
        y: Future series tensor of shape (out_steps, n_vertex, 3)
           Features: [traffic_value, time_of_day, day_of_week]
    """
    
    def __init__(
        self, 
        data: pd.DataFrame, 
        in_steps: int = 12,  # Not used, but kept for interface consistency
        out_steps: int = 12,
        steps_per_day: int = 288,
    ):
        # For encoder, we only need future series
        # We use in_steps=0 internally but still pass it for window calculation
        super().__init__(data, in_steps=in_steps, out_steps=out_steps, steps_per_day=steps_per_day)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """Get a single future series sample.
        
        Args:
            idx: Sample index
            
        Returns:
            y: Future series tensor of shape (out_steps, n_vertex, 3)
        """
        t = self.valid_indices[idx]
        y = self._get_y_features(t, include_temporal=True)
        return torch.from_numpy(y)


class PredictorDataset(TASSGNBaseDataset):
    """Dataset for Phase 3: Predictor training.
    
    Predictor learns to predict future pattern labels from history series.
    This dataset provides (x, cluster_labels) pairs.
    
    Args:
        data: Traffic data DataFrame
        cluster_labels: Cluster labels array of shape (num_samples, n_vertex, 1)
                       Generated in Phase 2 by Labeler
        in_steps: Number of input time steps (default: 12)
        out_steps: Number of output time steps (default: 12)
        steps_per_day: Number of time steps per day (default: 288)
        
    Returns:
        x: History series tensor of shape (in_steps, n_vertex, 3)
        labels: Cluster labels tensor of shape (n_vertex, 1)
    """
    
    def __init__(
        self, 
        data: pd.DataFrame, 
        cluster_labels: np.ndarray,
        in_steps: int = 12,
        out_steps: int = 12,
        steps_per_day: int = 288,
    ):
        super().__init__(data, in_steps=in_steps, out_steps=out_steps, steps_per_day=steps_per_day)
        
        # Validate cluster_labels shape
        if len(cluster_labels) != len(self.valid_indices):
            logger.warning(f"cluster_labels length ({len(cluster_labels)}) != valid_indices length ({len(self.valid_indices)})")
            # Assume cluster_labels corresponds to sequential samples
            # and adjust if needed
        
        self.cluster_labels = cluster_labels.astype(np.int64)
        logger.info(f"PredictorDataset: {len(self)} samples with cluster labels")
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single (x, label) pair.
        
        Args:
            idx: Sample index
            
        Returns:
            x: History series tensor of shape (in_steps, n_vertex, 3)
            labels: Cluster labels tensor of shape (n_vertex, 1)
        """
        t = self.valid_indices[idx]
        x = self._get_x_features(t)
        labels = self.cluster_labels[idx]  # (n_vertex, 1)
        
        return torch.from_numpy(x), torch.from_numpy(labels)


class TASSGNDataset(TASSGNBaseDataset):
    """Dataset for Phase 5: Final TASSGN model training.
    
    TASSGN takes history series (x), self-sampled similar patterns (sample_data),
    and predicts future series (y).
    
    Args:
        data: Traffic data DataFrame
        sample_index: Self-sampling indices array of shape (num_samples, n_vertex, num_samples_per_node, 1)
                     Generated in Phase 4
        window_data: All window data for sampling, shape (total_windows, out_steps, n_vertex, 3)
        in_steps: Number of input time steps (default: 12)
        out_steps: Number of output time steps (default: 12)
        steps_per_day: Number of time steps per day (default: 288)
        num_samples: Number of samples per node (default: 7)
        
    Returns:
        x: History series tensor of shape (in_steps, n_vertex, 3)
        sample_data: Sampled similar patterns tensor of shape (num_samples, out_steps, n_vertex, 3)
        y: Future series tensor of shape (out_steps, n_vertex, 1) for prediction target
    """
    
    def __init__(
        self, 
        data: pd.DataFrame, 
        sample_index: np.ndarray,
        window_data: np.ndarray,
        in_steps: int = 12,
        out_steps: int = 12,
        steps_per_day: int = 288,
        num_samples: int = 7,
    ):
        super().__init__(data, in_steps=in_steps, out_steps=out_steps, steps_per_day=steps_per_day)
        
        self.num_samples = num_samples
        self.sample_index = sample_index  # (num_samples, n_vertex, num_samples_per_node, 1)
        
        # Reshape window_data for efficient indexing
        # Original: (total_windows, out_steps, n_vertex, features)
        # Flattened: (total_windows * n_vertex, out_steps, features)
        self.window_data = window_data
        total_windows, out_len, n_nodes, features = window_data.shape
        self.window_data_flat = window_data.transpose(0, 2, 1, 3).reshape(-1, out_len, features)
        # Now shape: (total_windows * n_vertex, out_steps, features)
        
        logger.info(f"TASSGNDataset: {len(self)} samples, {num_samples} samples per node")
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get a single (x, sample_data, y) triplet.
        
        Args:
            idx: Sample index
            
        Returns:
            x: History series tensor of shape (in_steps, n_vertex, 3)
            sample_data: Sampled patterns tensor of shape (num_samples, out_steps, n_vertex, 3)
            y: Future series tensor of shape (out_steps, n_vertex, 1)
        """
        t = self.valid_indices[idx]
        
        # Get input features
        x = self._get_x_features(t)
        
        # Get target (only traffic values for loss computation)
        y = self._get_y_features(t, include_temporal=False)
        
        # Get sampled similar patterns
        sample_idx = self.sample_index[idx]  # (n_vertex, num_samples, 1)
        sample_idx_flat = sample_idx.reshape(-1)  # (n_vertex * num_samples,)
        
        # Index into flattened window data
        sample_data_flat = self.window_data_flat[sample_idx_flat, :, :]
        # Shape: (n_vertex * num_samples, out_steps, features)
        
        # Reshape to (n_vertex, num_samples, out_steps, features)
        sample_data = sample_data_flat.reshape(self.n_vertex, self.num_samples, self.out_steps, -1)
        
        # Permute to (num_samples, out_steps, n_vertex, features)
        sample_data = sample_data.transpose(1, 2, 0, 3)
        
        return (
            torch.from_numpy(x),
            torch.from_numpy(sample_data.astype(np.float32)),
            torch.from_numpy(y)
        )


class WindowDataset(TASSGNBaseDataset):
    """Utility Dataset for generating window data (used in Phase 4).
    
    This dataset generates all possible future windows for self-sampling.
    
    Args:
        data: Traffic data DataFrame
        out_steps: Length of future series (default: 12)
        steps_per_day: Number of time steps per day (default: 288)
        
    Returns:
        window: Future window tensor of shape (out_steps, n_vertex, 3)
    """
    
    def __init__(
        self, 
        data: pd.DataFrame, 
        out_steps: int = 12,
        steps_per_day: int = 288,
    ):
        # Use 0 in_steps since we only need future windows
        super().__init__(data, in_steps=0, out_steps=out_steps, steps_per_day=steps_per_day)
        
        # Recompute valid indices for window-only dataset
        self.valid_indices = self._compute_window_indices()
        
        logger.info(f"WindowDataset: {len(self)} windows")
    
    def _compute_window_indices(self) -> np.ndarray:
        """Compute valid window indices (allow some NaN for completeness)."""
        num_possible = len(self.data_values) - self.out_steps + 1
        
        if num_possible <= 0:
            return np.array([], dtype=np.int64)
        
        # For window data, we include all possible windows
        # NaN handling is done during sampling
        return np.arange(num_possible, dtype=np.int64)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """Get a single window.
        
        Args:
            idx: Window index
            
        Returns:
            window: Future window tensor of shape (out_steps, n_vertex, 3)
        """
        t = idx  # Direct index for window dataset
        
        window = np.zeros((self.out_steps, self.n_vertex, 3), dtype=np.float32)
        window[:, :, 0] = self.scaled_data[t:t + self.out_steps, :]
        window[:, :, 1] = self.tod[t:t + self.out_steps, np.newaxis]
        window[:, :, 2] = self.dow[t:t + self.out_steps, np.newaxis]
        
        return torch.from_numpy(window)
    
    def get_all_windows(self) -> np.ndarray:
        """Get all windows as a numpy array.
        
        Returns:
            Array of shape (num_windows, out_steps, n_vertex, 3)
        """
        num_windows = len(self)
        windows = np.zeros((num_windows, self.out_steps, self.n_vertex, 3), dtype=np.float32)
        
        for i in range(num_windows):
            t = i
            windows[i, :, :, 0] = self.scaled_data[t:t + self.out_steps, :]
            windows[i, :, :, 1] = self.tod[t:t + self.out_steps, np.newaxis]
            windows[i, :, :, 2] = self.dow[t:t + self.out_steps, np.newaxis]
        
        return windows
