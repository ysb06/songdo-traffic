from typing import Callable, List, Literal, Optional, Tuple, Union
import logging

import lightning as L
import numpy as np
import pandas as pd
from metr.components import TrafficData
from sklearn.preprocessing import MinMaxScaler
from torch import FloatTensor, Tensor, device
import torch
from torch.utils.data import DataLoader, Dataset, Subset, ConcatDataset
from tqdm import tqdm

logger = logging.getLogger(__name__)

CallateFnInput = List[Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex, pd.Timestamp]]


class TrafficCoreDataset(Dataset):
    def __init__(self, data: np.ndarray, seq_length: int = 24, allow_nan: bool = False):
        super().__init__()
        if len(data) < seq_length:
            raise ValueError("Data length should be larger than seq_length")

        self.seq_length = seq_length
        self.data = data.reshape(-1, 1)
        self.scaled_data = self.data

        # self.cursors: List[int] = []
        # if allow_nan == False:
        #     for i in range(len(data) - seq_length):
        #         x, y, _, _ = self._getitem(i)
        #         if not np.isnan(x).any() and not np.isnan(y).any():
        #             self.cursors.append(i)
        # else:
        #     self.cursors = list(range(len(data) - seq_length))

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


class TrafficDataset(TrafficCoreDataset):
    def __init__(
        self,
        data: pd.Series,
        seq_length: int = 24,
        allow_nan: bool = False,
    ):
        super().__init__(data.to_numpy(), seq_length, allow_nan)
        self.data_df = data

    def __getitem__(
        self,
        index: int,
    ) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex, pd.Timestamp]:
        x, y, x_idxs, y_idx = super().__getitem__(index)

        x_time_indices = self.data_df.index[x_idxs]  # x_idxs = range(i, i+seq_length)
        y_time_index = self.data_df.index[y_idx]  # y_idx = i + seq_length

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

        # valid_indices: List[int] = []
        # for dataset_idx, i in enumerate(self.cursors):
        #     label_time = self.data_df.index[i + self.seq_length]
        #     if start_ts <= label_time <= end_ts - pd.Timedelta(hours=self.seq_length):
        #         valid_indices.append(dataset_idx)

        return Subset(self, valid_indices)


def collate_fn_all(
    batch: CallateFnInput,
) -> Tuple[Tensor, Tensor, List[pd.DatetimeIndex], List[pd.Timestamp]]:
    xs = [item[0] for item in batch]
    ys = [item[1] for item in batch]
    x_time_indices = [item[2] for item in batch]
    y_time_indices = [item[3] for item in batch]

    xs_t = [torch.from_numpy(x).float() for x in xs]
    ys_t = [torch.from_numpy(y).float() for y in ys]

    xs_t = torch.stack(xs_t, dim=0)
    ys_t = torch.stack(ys_t, dim=0)

    return xs_t, ys_t, x_time_indices, y_time_indices


def collate_fn_simple(batch: CallateFnInput) -> Tuple[Tensor, Tensor]:
    xs = [item[0] for item in batch]
    ys = [item[1] for item in batch]

    xs_t = [torch.from_numpy(x).float() for x in xs]
    ys_t = [torch.from_numpy(y).float() for y in ys]

    xs_t = torch.stack(xs_t, dim=0)
    ys_t = torch.stack(ys_t, dim=0)

    return xs_t, ys_t


class BaseTrafficDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_path: str = "../datasets/metr-imc/metr-imc.h5",
        seq_length: int = 24,
        batch_size: int = 64,
        num_workers: int = 1,
        collate_fn: Callable[[CallateFnInput], Tuple] = collate_fn_all,
        valid_split_datetime: Optional[str] = "2024-09-01 00:00:00",
        test_target_sensor: List[str] = [],
        strict_scaling: bool = False,
    ):
        super().__init__()
        self.data_path = data_path
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.collate_fn = collate_fn
        self.valid_split_datetime = valid_split_datetime
        self.test_target_sensor = test_target_sensor

        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.strict_scaler = strict_scaling

        # Setup 단계에서 로드 및 초기화
        self.raw_data: Optional[pd.DataFrame] = None
        self.train_dataset: Optional[ConcatDataset] = None
        self.valid_dataset: Optional[ConcatDataset] = None
        self.test_dataset: Optional[ConcatDataset] = None

    def setup(self, _: Optional[Literal["fit", "validate", "test", "predict"]] = None):
        logger.info(f"Setting up data module")
        logger.info(f"Loading data from {self.data_path}")
        self.raw_data = TrafficData.import_from_hdf(self.data_path).data

        # Train, Valid, Test Dataset Split 전략
        # Train과 Test는 센서(열) 기준으로 분리
        all_sensor_list = self.raw_data.columns
        test_sensor_names = set(self.test_target_sensor)

        logger.info(f"Splitting to test datasets...")
        all_datasets: List[TrafficDataset] = []
        train_valid_datasets: List[TrafficDataset] = []
        test_datasets: List[TrafficDataset] = []
        for sensor_name in tqdm(all_sensor_list):
            sensor_data = self.raw_data[sensor_name]
            dataset = TrafficDataset(sensor_data, seq_length=self.seq_length)
            if not len(dataset) > 0:
                continue

            all_datasets.append(dataset)
            if sensor_name in test_sensor_names:
                test_datasets.append(dataset)
            else:
                train_valid_datasets.append(dataset)

        # Train과 Valid는 시간 기준으로 분리
        # 마지막 달이 Valid 데이터
        logger.info(f"Splitting to valid datasets...")
        train_datasets = []
        valid_datasets = []
        if self.valid_split_datetime is not None:
            for dataset in tqdm(train_valid_datasets, position=0):
                train_subset = dataset.get_subset(
                    end_datetime=self.valid_split_datetime
                )
                valid_subset = dataset.get_subset(
                    start_datetime=self.valid_split_datetime
                )

                if len(train_subset) > 0:
                    train_datasets.append(train_subset)
                if len(valid_subset) > 0:
                    valid_datasets.append(valid_subset)
        else:
            train_datasets = train_valid_datasets

        # 모두 연결
        # 이상치는 모두 제거된 상태여야 함
        logger.info(f"Concatenating datasets...")
        self.train_dataset = ConcatDataset(train_datasets)
        if len(valid_datasets) > 0:
            self.valid_dataset = ConcatDataset(valid_datasets)
        if len(test_datasets) > 0:
            self.test_dataset = ConcatDataset(test_datasets)

        # Scaler 적용
        logger.info(f"Fitting scaler...")
        if self.strict_scaler:
            # 엄밀하게는 이렇게 해야하지만, 너무 오래 걸림
            x_list = []
            y_list = []
            for x, y, *_ in tqdm(self.train_dataset):
                x_list.append(x)
                y_list.append(y.reshape(1, 1))
            scaler_ref = np.concatenate(x_list + y_list, axis=0).reshape(-1, 1)
            self.scaler.fit(scaler_ref)
        else:
            train_sensor_names = set(all_sensor_list) - test_sensor_names
            scaling_target: pd.DataFrame = self.raw_data.loc[:, list(train_sensor_names)]
            if self.valid_split_datetime is not None:
                scaling_target = scaling_target.loc[: self.valid_split_datetime]
            scaling_target = scaling_target.to_numpy().reshape(-1, 1)
            self.scaler.fit(scaling_target)

        logger.info(f"Applying scaler...")
        for dataset in all_datasets:
            dataset.apply_scaler(self.scaler)
        
        logger.info(f"Setup complete")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )
