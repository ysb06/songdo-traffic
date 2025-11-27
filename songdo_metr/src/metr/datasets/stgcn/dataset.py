from typing import Optional, Protocol, runtime_checkable
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler

class STGCNDataset(Dataset):
    """PyTorch Dataset for STGCN model.
    
    Args:
        x: Input tensor of shape (num_samples, in_channels, n_vertex, n_his)
        y: Target tensor of shape (num_samples, n_vertex)
    """
    
    def __init__(self, data: np.ndarray, n_his: int, n_pred: int):
        self.x, self.y = self._data_transform(data, n_his, n_pred)
        
        assert len(self.x) == len(self.y), "x and y must have the same length"
    
    def __len__(self) -> int:
        return len(self.x)
    
    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]
    
    def _data_transform(data: np.ndarray, n_his: int, n_pred: int):
        """Transform time series data into STGCN input format.

        Args:
            data: numpy array of shape (time_steps, n_vertex)
            n_his: number of historical time steps
            n_pred: number of prediction time steps ahead

        Returns:
            x: torch.Tensor of shape (num_samples, 1, n_vertex, n_his)
            y: torch.Tensor of shape (num_samples, n_vertex)
        """
        n_vertex = data.shape[1]
        l = len(data)
        num = l - n_his - n_pred

        # Shape: (num_samples, in_channels, n_vertex, n_his)
        x = np.zeros([num, 1, n_vertex, n_his], dtype=np.float32)
        y = np.zeros([num, n_vertex], dtype=np.float32)

        for i in range(num):
            head = i
            tail = i + n_his
            # Transpose to get (n_vertex, n_his) shape
            x[i, 0, :, :] = data[head:tail].T
            # Target is n_pred steps ahead from the last historical time step
            y[i] = data[tail + n_pred - 1]

        return torch.Tensor(x), torch.Tensor(y)
