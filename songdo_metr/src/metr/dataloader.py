import logging
from typing import Callable, List, Literal, Optional, Tuple

import lightning as L
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch import Tensor
from torch.utils.data import ConcatDataset, DataLoader, Subset
from tqdm import tqdm

from metr.datasets import TrafficDataset, TrafficDataType

logger = logging.getLogger(__name__)

CollateFnInput = List[TrafficDataType]


def collate_all(
    batch: CollateFnInput,
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


def collate_simple(batch: CollateFnInput) -> Tuple[Tensor, Tensor]:
    xs = [item[0] for item in batch]
    ys = [item[1] for item in batch]

    xs_t = [torch.from_numpy(x).float() for x in xs]
    ys_t = [torch.from_numpy(y).float() for y in ys]

    xs_t = torch.stack(xs_t, dim=0)
    ys_t = torch.stack(ys_t, dim=0)

    return xs_t, ys_t


class TrafficDataModule(L.LightningDataModule):
    def __init__(
        self,
        training_df: pd.DataFrame,
        test_df: pd.DataFrame,
        seq_length: int = 24,
        batch_size: int = 64,
        num_workers: int = 1,
        shuffle_training: bool = True,
        scaler: MinMaxScaler = MinMaxScaler(feature_range=(0, 1)),
        collate_fn: Callable[[CollateFnInput], Tuple] = collate_simple,
        valid_split_datetime: Optional[str] = "2024-08-01 00:00:00",
        training_target_sensor: Optional[List[str]] = None,
        scale_strictly: bool = False,
    ):
        super().__init__()
        self.training_df_raw = training_df
        self.test_df_raw = test_df

        self.seq_length = seq_length
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.shuffle_training = shuffle_training

        self.collate_fn = collate_fn
        self.valid_split_datetime = valid_split_datetime
        self.training_target_sensor = training_target_sensor

        self.scaler = scaler
        self.scale_strictly = scale_strictly

        # Setup 단계에서 로드 및 초기화
        self.train_dataset: Optional[ConcatDataset] = None
        self.valid_dataset: Optional[ConcatDataset] = None
        self.test_dataset: Optional[ConcatDataset] = None

    def _create_datasets(self, df: pd.DataFrame) -> List[TrafficDataset]:
        datasets: List[TrafficDataset] = []
        for sensor_name, sensor_data in tqdm(df.items(), total=len(df.columns), desc="Creating datasets..."):
            dataset = TrafficDataset(
                sensor_data,
                seq_length=self.seq_length,
                allow_nan=False,
            )
            if not len(dataset) > 0:
                logger.warning(f"Dataset {sensor_name} is empty")
                continue
            datasets.append(dataset)

        return datasets

    def _split_datasets(self, datasets: List[TrafficDataset], split_datetime: str):
        train_subsets: List[Subset] = []
        valid_subsets: List[Subset] = []

        for dataset in datasets:
            train_subset = dataset.get_subset(end_datetime=split_datetime)
            valid_subset = dataset.get_subset(start_datetime=split_datetime)
            if len(train_subset) > 0:
                train_subsets.append(train_subset)
            if len(valid_subset) > 0:
                valid_subsets.append(valid_subset)

        return train_subsets, valid_subsets

    def _get_strict_scaler_ref(self, dataset: TrafficDataset):
        x_list = []
        y_list = []
        for x, y, *_ in tqdm(dataset, desc="Creating strict scaler reference..."):
            x_list.append(x)
            y_list.append(y.reshape(1, 1))
        scaler_ref = np.concatenate(x_list + y_list, axis=0).reshape(-1, 1)

        return scaler_ref

    def _get_normal_scaler_ref(self, ref_raw_df: pd.DataFrame, end_datetime: str):
        scaling_target: pd.DataFrame = ref_raw_df.loc[:, ref_raw_df.columns]
        if self.valid_split_datetime is not None:
            scaling_target = scaling_target.loc[:end_datetime]

        return scaling_target.to_numpy().reshape(-1, 1)

    def setup(
        self,
        stage: Optional[Literal["fit", "validate", "test", "predict"]] = None,
    ):
        logger.info(f"Setting up data module in {stage} stage")

        # 데이터셋 생성
        # training_df_raw에서 training_target_sensor에 해당하는 데이터만 사용
        # training_target_sensor가 None이면 전체 데이터 사용
        training_targets: pd.DataFrame = self.training_df_raw
        if self.training_target_sensor is not None:
            training_targets = self.training_df_raw.loc[:, self.training_target_sensor]

        all_datasets: List[TrafficDataset] = []
        # Train, Valid 데이터셋 생성
        training_datasets = self._create_datasets(training_targets)
        all_datasets.extend(training_datasets)
        # Test 데이터셋 생성
        test_datasets = self._create_datasets(self.test_df_raw)
        all_datasets.extend(test_datasets)

        # Train과 Valid는 시간 기준으로 분리
        # 마지막 달이 Valid 데이터
        logger.info(f"Splitting to valid datasets...")
        training_subsets = []
        validation_subsets = []
        if self.valid_split_datetime is not None:
            training_subsets, validation_subsets = self._split_datasets(
                training_datasets,
                self.valid_split_datetime,
            )
        else:
            training_subsets = training_datasets

        # 모두 연결
        # 이상치는 모두 제거된 상태여야 함
        logger.info(f"Concatenating datasets...")
        self.train_dataset = ConcatDataset(training_subsets)
        if len(validation_subsets) > 0:
            self.valid_dataset = ConcatDataset(validation_subsets)
        if len(test_datasets) > 0:
            self.test_dataset = ConcatDataset(test_datasets)

        # Scaler 적용
        logger.info(f"Fitting scaler...")
        ref_data: Optional[np.ndarray] = None
        if self.scale_strictly:
            # 엄밀하게는 이렇게 해야하지만, 너무 오래 걸림
            ref_data = self._get_strict_scaler_ref(self.train_dataset)
        else:
            end_datetime = self.valid_split_datetime
            if self.valid_split_datetime is None:
                end_datetime = training_targets.index[-1]
            ref_data = self._get_normal_scaler_ref(self.training_df_raw, end_datetime)
        self.scaler.fit(ref_data)

        logger.info(f"Applying scaler...")
        for dataset in all_datasets:
            dataset.apply_scaler(self.scaler)

        logger.info(f"Setup complete")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle_training,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn
        )

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn
        )
