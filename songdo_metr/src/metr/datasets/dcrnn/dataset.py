from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class DCRNNDataset(Dataset):
    """PyTorch Dataset for DCRNN model.

    Generates seq2seq style input/output pairs with optional time features.

    Args:
        data: Traffic data DataFrame with DatetimeIndex, shape (time_steps, num_nodes)
        seq_len: Number of historical time steps (default: 12)
        horizon: Number of prediction time steps (default: 12)
        add_time_in_day: Whether to add time-of-day feature (default: True)
        add_day_in_week: Whether to add day-of-week feature (default: False)

    Returns from __getitem__:
        x: Input tensor of shape (seq_len, num_nodes, input_dim)
           where input_dim = 1 + (1 if add_time_in_day) + (7 if add_day_in_week)
        y: Target tensor of shape (horizon, num_nodes, output_dim)
           where output_dim = 1 (traffic value only)
    """

    def __init__(
        self,
        data: pd.DataFrame,
        seq_len: int = 12,
        horizon: int = 12,
        add_time_in_day: bool = True,
        add_day_in_week: bool = False,
    ):
        self.seq_len = seq_len
        self.horizon = horizon
        self.add_time_in_day = add_time_in_day
        self.add_day_in_week = add_day_in_week

        # Calculate input_dim based on features
        self.input_dim = 1  # traffic value
        if add_time_in_day:
            self.input_dim += 1
        if add_day_in_week:
            self.input_dim += 7

        self.output_dim = 1  # predict traffic value only

        self.x, self.y = self._data_transform(data)

        assert len(self.x) == len(self.y), "x and y must have the same length"

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]

    def _data_transform(
        self, df: pd.DataFrame
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Transform time series data into DCRNN seq2seq input format.

        Args:
            df: DataFrame with DatetimeIndex, shape (time_steps, num_nodes)

        Returns:
            x: torch.Tensor of shape (num_samples, seq_len, num_nodes, input_dim)
            y: torch.Tensor of shape (num_samples, horizon, num_nodes, output_dim)
        """
        num_samples, num_nodes = df.shape

        # Start with traffic data: (time_steps, num_nodes, 1)
        traffic_data = np.expand_dims(df.values, axis=-1)
        data_list = [traffic_data]

        # Add time-of-day feature: normalized to [0, 1]
        if self.add_time_in_day:
            time_ind = (
                df.index.values - df.index.values.astype("datetime64[D]")
            ) / np.timedelta64(1, "D")
            # Shape: (time_steps,) -> (time_steps, num_nodes, 1)
            time_in_day = np.tile(time_ind.reshape(-1, 1, 1), [1, num_nodes, 1])
            data_list.append(time_in_day)

        # Add day-of-week feature: one-hot encoding (7 dimensions)
        if self.add_day_in_week:
            day_in_week = np.zeros((num_samples, num_nodes, 7), dtype=np.float32)
            day_indices = pd.to_datetime(df.index).dayofweek.values
            for t in range(num_samples):
                day_in_week[t, :, day_indices[t]] = 1.0
            data_list.append(day_in_week)

        # Concatenate all features: (time_steps, num_nodes, input_dim)
        data = np.concatenate(data_list, axis=-1).astype(np.float32)

        # Generate sliding window samples
        # x_offsets: [-11, -10, ..., -1, 0] (past 12 steps including current)
        # y_offsets: [1, 2, ..., 12] (future 12 steps)
        x_offsets = np.arange(-self.seq_len + 1, 1)  # [-11, -10, ..., 0]
        y_offsets = np.arange(1, self.horizon + 1)  # [1, 2, ..., 12]

        min_t = abs(min(x_offsets))  # 11
        max_t = num_samples - max(y_offsets)  # num_samples - 12

        x_list, y_list = [], []
        for t in range(min_t, max_t):
            # x: (seq_len, num_nodes, input_dim)
            x_t = data[t + x_offsets, :, :]
            # y: (horizon, num_nodes, output_dim) - traffic value only
            y_t = data[t + y_offsets, :, :1]  # only first feature (traffic)
            x_list.append(x_t)
            y_list.append(y_t)

        x = np.stack(x_list, axis=0)  # (num_samples, seq_len, num_nodes, input_dim)
        y = np.stack(y_list, axis=0)  # (num_samples, horizon, num_nodes, output_dim)

        return torch.from_numpy(x), torch.from_numpy(y)

    @property
    def num_nodes(self) -> int:
        """Return the number of nodes (sensors)."""
        return self.x.shape[2]
