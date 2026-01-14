from typing import Optional, Protocol, runtime_checkable
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler

class STGCNDataset(Dataset):
    """PyTorch Dataset for STGCN model.
    
    Args:
        data: Traffic data array of shape (time_steps, n_vertex)
        n_his: Number of historical time steps
        n_pred: Number of prediction time steps ahead
        
    Returns:
        x: Input tensor of shape (num_samples, in_channels, n_his, n_vertex)
           where in_channels=1 for univariate traffic speed/flow
        y: Target tensor of shape (num_samples, n_vertex)
    """
    
    def __init__(self, data: np.ndarray, n_his: int, n_pred: int):
        self.x, self.y = self._data_transform(data, n_his, n_pred)
        self._is_scaled = False
        
        assert len(self.x) == len(self.y), "x and y must have the same length"
        if len(self.x) == 0:
            raise ValueError("All samples contained NaNs and were filtered out. Check your data.")
    
    def apply_scaler(self, scaler: MinMaxScaler) -> None:
        """Apply scaling to the dataset using a fitted scaler.
        
        Args:
            scaler: Fitted MinMaxScaler instance
        """
        if self._is_scaled:
            return
        
        # Skip scaling if dataset is empty
        if len(self.x) == 0:
            self._is_scaled = True
            return
        
        # Scale x: shape (num_samples, in_channels, n_his, n_vertex)
        x_shape = self.x.shape
        x_flat = self.x.numpy().reshape(-1, 1)
        x_scaled = scaler.transform(x_flat)
        self.x = torch.tensor(x_scaled.reshape(x_shape), dtype=torch.float32)
        
        # Scale y: shape (num_samples, n_vertex)
        y_shape = self.y.shape
        y_flat = self.y.numpy().reshape(-1, 1)
        y_scaled = scaler.transform(y_flat)
        self.y = torch.tensor(y_scaled.reshape(y_shape), dtype=torch.float32)
        
        self._is_scaled = True
    
    def __len__(self) -> int:
        return len(self.x)
    
    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]
    
    def _data_transform(self, data: np.ndarray, n_his: int, n_pred: int):
        n_vertex = data.shape[1]
        l = len(data)
        num = l - n_his - n_pred

        # Shape: (num_samples, in_channels, n_his, n_vertex)
        # Changed from (num_samples, in_channels, n_vertex, n_his) to match model expectation
        x_list, y_list = [], []
        filtered_count = 0

        for i in range(num):
            head = i
            tail = i + n_his
            
            # 1. 입력(x) 및 타겟(y) 구간 추출
            # Extract historical window: shape (n_his, n_vertex)
            # Directly assign without transpose to get (n_his, n_vertex) in the last two dimensions
            x_window = data[head:tail, :]  # (n_his, n_vertex)
            # Target is n_pred steps ahead from the last historical time step
            y_window = data[tail + n_pred - 1]  # (n_vertex,)

            # 2. 결측치 존재 여부 확인 (Any NaN in window)
            if np.isnan(x_window).any() or np.isnan(y_window).any():
                filtered_count += 1
                continue # NaN이 하나라도 있으면 해당 시점의 샘플은 건너뜀

            # 3. 모델 기대 형식으로 변형하여 리스트 추가
            x_list.append(torch.tensor(x_window).unsqueeze(0)) # (1, n_his, n_vertex)
            y_list.append(torch.tensor(y_window))

        # 리스트를 하나의 텐서로 결합
        x = torch.stack(x_list) if x_list else torch.empty(0) # (num_samples, 1, n_his, n_vertex)
        y = torch.stack(y_list) if y_list else torch.empty(0) # (num_samples, n_vertex)
        
        # Log filtering statistics
        if num > 0:
            retention_rate = len(x_list) / num * 100
            print(f"Dataset filtering: {len(x_list)}/{num} samples retained ({retention_rate:.1f}%), {filtered_count} filtered due to NaNs")

        return x, y