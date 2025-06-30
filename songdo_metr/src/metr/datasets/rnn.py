import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, Subset

logger = logging.getLogger(__name__)


class TrafficCoreDataset(Dataset):
    """Core dataset class for time-series traffic data. Creates sequences from 1D traffic data for RNN training."""

    def __init__(self, data: np.ndarray, seq_length: int = 24, allow_nan: bool = False):
        """Core dataset class for time-series traffic data. Creates sequences from 1D traffic data for RNN training.

        Args:
            data (np.ndarray): 1D array of traffic values
            seq_length (int, optional): Length of input sequences. Defaults to 24.
            allow_nan (bool, optional): Whether to allow NaN values in sequences. Defaults to False.
        """
        super().__init__()
        if len(data) < seq_length:
            raise ValueError("Data length should be larger than seq_length")

        self.seq_length = seq_length
        self.data = data.reshape(-1, 1)
        self.scaled_data = self.data

        self.cursors: List[int] = []
        time_len = len(self.data)
        if not allow_nan:
            isnan_arr = np.isnan(self.data).reshape(-1)
            cumsum = np.cumsum(isnan_arr, dtype=np.int32)
            cumsum = np.insert(cumsum, 0, 0)
            all_i = np.arange(time_len - seq_length)
            x_nan_count = cumsum[all_i + seq_length] - cumsum[all_i]
            y_is_nan = isnan_arr[all_i + seq_length]
            valid_mask = (x_nan_count == 0) & (~y_is_nan)
            valid_i = all_i[valid_mask]
            self.cursors = valid_i.tolist()
        else:
            self.cursors = list(range(time_len - seq_length))

    def __len__(self) -> int:
        return len(self.cursors)

    def __getitem__(self, index: int):
        return self._getitem(self.cursors[index])

    def _getitem(self, index: int) -> Tuple[np.ndarray, np.ndarray, List[int], int]:
        x = self.scaled_data[index : index + self.seq_length]
        y = self.scaled_data[index + self.seq_length]

        x_idxs = list(range(index, index + self.seq_length))
        y_idx = index + self.seq_length

        return x, y, x_idxs, y_idx

    def apply_scaler(self, scaler: MinMaxScaler):
        self.scaled_data = scaler.transform(self.data)


class BaseTrafficDataset(TrafficCoreDataset):
    """Core dataset class for time-series traffic data with time index support. Creates sequences from 1D traffic data for RNN training."""

    def __init__(
        self,
        data: pd.Series,
        seq_length: int = 24,
        allow_nan: bool = False,
    ):
        """Core dataset class for time-series traffic data with time index support. Creates sequences from 1D traffic data for RNN training.

        Args:
            data (pd.Series): Time-indexed pandas Series of traffic values
            seq_length (int, optional): Length of input sequences. Defaults to 24.
            allow_nan (bool, optional): Whether to allow NaN values in sequences. Defaults to False.
        """
        super().__init__(data.to_numpy(), seq_length, allow_nan)
        self.data_df = data

    def __getitem__(
        self,
        index: int,
    ) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex, pd.Timestamp]:
        x, y, x_idxs, y_idx = super().__getitem__(index)

        x_time_indices = self.data_df.index[x_idxs]
        y_time_index = self.data_df.index[y_idx]

        return x, y, x_time_indices, y_time_index

    @property
    def name(self) -> str:
        return self.data_df.name

    def get_subset(
        self,
        start_datetime: Optional[str] = None,
        end_datetime: Optional[str] = None,
    ) -> Subset:
        if start_datetime is None:
            start_ts = self.data_df.index[0]
        else:
            start_ts = pd.to_datetime(start_datetime)

        if end_datetime is None:
            end_ts = self.data_df.index[-1]
        else:
            end_ts = pd.to_datetime(end_datetime)

        label_times = self.data_df.index[[c + self.seq_length for c in self.cursors]]
        mask = (label_times >= start_ts) & (
            label_times <= end_ts - pd.Timedelta(hours=self.seq_length)
        )
        valid_indices = np.where(mask)[0].tolist()

        return Subset(self, valid_indices)
