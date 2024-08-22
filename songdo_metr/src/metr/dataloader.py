import torch
from torch.utils.data import Dataset, DataLoader, Subset
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Union


class MetrDataset(Dataset):
    @staticmethod
    def from_file(
        file_path: str, history_len: int, prediction_len: int
    ) -> "MetrDataset":
        return MetrDataset(pd.read_hdf(file_path), history_len, prediction_len)

    @staticmethod
    def collate_fn(batch):
        x_batch = torch.stack([item[0] for item in batch])
        y_batch = torch.stack([item[1] for item in batch])
        return x_batch, y_batch

    def __init__(
        self,
        data_df: pd.DataFrame,
        history_len: int,
        prediction_len: int,
        device: Union[str, torch.device] = torch.device("cpu"),
    ) -> None:
        super().__init__()
        self.raw_df = data_df
        self.num_samples, self.num_nodes = self.raw_df.shape
        self.history_len = history_len
        self.prediction_len = prediction_len
        self.device = device

        # 스케일러 적용
        scaled_data = StandardScaler().fit_transform(self.raw_df)
        self.data = torch.tensor(scaled_data, dtype=torch.float32).to(self.device)

    def __len__(self) -> int:
        return self.num_samples - self.history_len - self.prediction_len

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.data[index : index + self.history_len]
        y = self.data[index + self.history_len + self.prediction_len - 1]
        return x.unsqueeze(0), y

    def split(
        self, train_ratio: float = 0.7, valid_ratio: float = 0.1, allow_overlap: bool = False
    ) -> Tuple[Subset, Subset, Subset]:
        """Generates subsets for training, validation, and testing. The test ratio is automatically calculated.

        Args:
            train_ratio (float): The proportion of the dataset to be used for training.
            val_ratio (float): The proportion of the dataset to be used for validation.
            allow_overlap (bool, optional): If True, allows overlapping between subsets, meaning that data in one subset
                                            might be included as input (x) in another. This is generally set to False
                                            to ensure independence between subsets. Defaults to False.

        Returns:
            Tuple[Subset, Subset, Subset]: The training, validation, and testing subsets.
        """

        len_train = round(self.num_samples * train_ratio)
        len_valid = round(self.num_samples * valid_ratio)

        total_indices = list(range(len(self)))
        offset = self.history_len + self.prediction_len if not allow_overlap else 0
        train_indices = total_indices[: len_train - offset]
        valid_indices = total_indices[len_train : len_train + len_valid - offset]
        test_indices = total_indices[len_train + len_valid :]

        train_df = self.raw_df.iloc[total_indices[:len_train]]
        scaler = StandardScaler().fit(train_df)
        self.data = torch.tensor(scaler.transform(self.raw_df), dtype=torch.float32).to(
            self.device
        )

        train_subset = Subset(self, train_indices)
        val_subset = Subset(self, valid_indices)
        test_subset = Subset(self, test_indices)

        return train_subset, val_subset, test_subset
