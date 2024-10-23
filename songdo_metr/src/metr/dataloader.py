import torch
from torch.utils.data import Dataset, DataLoader, Subset
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)


class MetrDataset(Dataset):
    @staticmethod
    def from_file(
        data_path: str,
        history_len: int,
        prediction_len: int,
        missing_value_path: Optional[str] = None,
    ) -> "MetrDataset":
        return MetrDataset(
            pd.read_hdf(data_path),
            history_len,
            prediction_len,
            pd.read_hdf(missing_value_path) if missing_value_path else None,
        )

    @staticmethod
    def collate_fn(batch):
        x_batch = torch.stack([item[0] for item in batch])
        y_batch = torch.stack([item[1] for item in batch])

        return x_batch, y_batch

    @staticmethod
    def collate_fn_with_missing(batch):
        x_batch = torch.stack([item[0] for item in batch])
        y_batch = torch.stack([item[1] for item in batch])
        # May error occur if missing_y_batch is None
        missing_y_batch = torch.stack([item[5] for item in batch])

        return x_batch, y_batch, missing_y_batch

    @staticmethod
    def collate_fn_debug(batch):
        x_batch = torch.stack([item[0] for item in batch])
        y_batch = torch.stack([item[1] for item in batch])
        raw_x_batch = torch.stack([item[2] for item in batch])
        raw_y_batch = torch.stack([item[3] for item in batch])
        missing_x_batch = torch.stack([item[4] for item in batch])
        missing_y_batch = torch.stack([item[5] for item in batch])

        return (
            x_batch,
            y_batch,
            raw_x_batch,
            raw_y_batch,
            missing_x_batch,
            missing_y_batch,
        )

    def __init__(
        self,
        data_df: pd.DataFrame,
        history_len: int,
        prediction_len: int,
        missing_df: Optional[pd.DataFrame] = None,
    ) -> None:
        super().__init__()
        self.raw_df = data_df
        self.num_samples, self.num_nodes = self.raw_df.shape
        self.history_len = history_len
        self.prediction_offset = prediction_len
        self.missing_df = missing_df
        if self.missing_df is None:
            logger.warning(
                "No missing value label. This may cause errors or incorrect calculations for some metrics."
            )

        # 스케일러 적용
        self.scaler_for_all = StandardScaler().fit(self.raw_df)
        logger.info(f"Fitting StandardScaler(All) Complete")
        scaled_data = self.scaler_for_all.transform(self.raw_df)
        self.raw_data = torch.tensor(self.raw_df.to_numpy(), dtype=torch.int32)
        self.data = torch.tensor(scaled_data, dtype=torch.float32)
        self.missings_data = (
            torch.tensor(self.missing_df.to_numpy())
            if self.missing_df is not None
            else None
        )

    def __len__(self) -> int:
        return self.num_samples - self.history_len - self.prediction_offset

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.data[index : index + self.history_len]
        y = self.data[index + self.history_len + self.prediction_offset - 1]
        raw_x = self.raw_data[index : index + self.history_len]
        raw_y = self.raw_data[index + self.history_len + self.prediction_offset - 1]
        if self.missing_df is None:
            return x.unsqueeze(0), y, raw_x.unsqueeze(0), raw_y
        else:
            missing_x = self.missings_data[index : index + self.history_len]
            missing_y = self.missings_data[
                index + self.history_len + self.prediction_offset - 1
            ]
            return x.unsqueeze(0), y, raw_x.unsqueeze(0), raw_y, missing_x, missing_y

    @property
    def is_missing_value_labeled(self):
        return self.missing_df is not None

    def split(
        self,
        train_ratio: float = 0.7,
        valid_ratio: float = 0.1,
        allow_training_overlap: bool = False,
        allow_test_overlap: bool = False,
    ) -> Tuple[Subset, Subset, Subset, StandardScaler]:
        """
        Splits the dataset into training, validation, and test subsets, with optional control over overlap between the subsets.

        Args:
            train_ratio (float): The proportion of the dataset to be used for training. Defaults to 0.7.
            valid_ratio (float): The proportion of the dataset to be used for validation. The test set ratio is automatically 
                                determined as the remaining portion after training and validation. Defaults to 0.1.
            allow_training_overlap (bool): If True, allows overlap between the end of the training data and the start of the 
                                        validation data. If False, ensures that the validation data starts after the history 
                                        and prediction window of the training data. Defaults to False.
            allow_test_overlap (bool): If True, allows overlap between the end of the validation data and the start of the 
                                    test data. If False, ensures that the test data starts after the history and prediction 
                                    window of the validation data. Defaults to False.

        Returns:
            Tuple[Subset,Subset,Subset,StandardScaler]: 
                - train_subset (Subset): The training subset of the dataset.
                - val_subset (Subset): The validation subset of the dataset.
                - test_subset (Subset): The test subset of the dataset.
                - split_scaler (StandardScaler): A scaler fitted on the training data, which is applied to the entire dataset.
        """

        len_train = round(self.num_samples * train_ratio)
        len_valid = round(self.num_samples * valid_ratio)

        total_indices = list(range(len(self)))
        valid_offset = self.history_len + self.prediction_offset if not allow_training_overlap else 0
        test_offset = self.history_len + self.prediction_offset if not allow_test_overlap else 0
        train_indices = total_indices[: len_train]
        valid_indices = total_indices[len_train + valid_offset : len_train + len_valid + valid_offset]
        test_indices = total_indices[len_train + len_valid + valid_offset + test_offset:]

        train_df = self.raw_df.iloc[total_indices[:len_train]]
        split_scaler = StandardScaler().fit(train_df)
        logger.info(f"Fitting StandardScaler(Training) Complete")
        self.data = torch.tensor(
            split_scaler.transform(self.raw_df), dtype=torch.float32
        )

        train_subset = Subset(self, train_indices)
        val_subset = Subset(self, valid_indices)
        test_subset = Subset(self, test_indices)

        return train_subset, val_subset, test_subset, split_scaler
