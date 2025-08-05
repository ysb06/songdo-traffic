import logging
from typing import Callable, List, Literal, Optional, Tuple

import lightning as L
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import ConcatDataset, DataLoader, Subset
from tqdm import tqdm

from metr.datasets.rnn.dataloader import collate_simple
from metr.datasets.rnn.dataset import TrafficDataset, TrafficDataType
from metr.components.metr_imc.traffic_data import get_raw

logger = logging.getLogger(__name__)

CollateFnInput = List[TrafficDataType]


class TrafficDataModule(L.LightningDataModule):
    def __init__(
        self,
        dataset_path: str,
        training_start_datetime: str = "2022-11-01 00:00:00",
        training_end_datetime: str = "2024-07-31 23:59:59",
        validation_start_datetime: str = "2024-08-01 00:00:00",
        validation_end_datetime: str = "2024-09-30 23:59:59",
        test_start_datetime: str = "2024-10-01 00:00:00",
        test_end_datetime: str = "2024-10-31 23:59:59",
        seq_length: int = 24,
        batch_size: int = 64,
        num_workers: int = 1,
        shuffle_training: bool = True,
        collate_fn: Callable[[CollateFnInput], Tuple] = collate_simple,
        training_target_sensors: Optional[List[str]] = None,
        test_target_sensors: Optional[List[str]] = None,
        scale_strictly: bool = False,
    ):
        super().__init__()
        self.dataset_path = dataset_path
        self.training_start_datetime = training_start_datetime
        self.training_end_datetime = training_end_datetime
        self.validation_start_datetime = validation_start_datetime
        self.validation_end_datetime = validation_end_datetime
        self.test_start_datetime = test_start_datetime
        self.test_end_datetime = test_end_datetime

        self.seq_length = seq_length
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.scale_strictly = scale_strictly
        self.scaler = MinMaxScaler(feature_range=(0, 1))

        self.shuffle_training = shuffle_training
        self.collate_fn = collate_fn

        self.training_target_sensors = training_target_sensors
        self.test_target_sensors = test_target_sensors

        self.train_dataset: Optional[ConcatDataset] = None
        self.valid_dataset: Optional[ConcatDataset] = None
        self.test_dataset: Optional[ConcatDataset] = None

    def _get_strict_scaler_ref(self, dataset: TrafficDataset) -> np.ndarray:
        x_list = []
        y_list = []
        for x, y, *_ in tqdm(dataset, desc="Creating strict scaler reference..."):
            x_list.append(x)
            y_list.append(y.reshape(1, 1))
        scaler_ref = np.concatenate(x_list + y_list, axis=0).reshape(-1, 1)

        return scaler_ref

    def _get_normal_scaler_ref(self, ref_df: pd.DataFrame, end_datetime: str):
        scaling_target: pd.DataFrame = ref_df.loc[:end_datetime]
        return scaling_target.to_numpy().reshape(-1, 1)

    def _create_sensor_datasets(
        self, df: pd.DataFrame, allow_nan: bool
    ) -> List[TrafficDataset]:
        datasets: List[TrafficDataset] = []
        for sensor_name, sensor_data in tqdm(
            df.items(),
            total=len(df.columns),
            desc="Creating datasets...",
        ):
            dataset = TrafficDataset(
                sensor_data,
                seq_length=self.seq_length,
                allow_nan=allow_nan,
            )
            if not len(dataset) > 0:
                logger.warning(f"Dataset {sensor_name} is empty")
                continue
            datasets.append(dataset)

        return datasets

    def _split_df(
        self,
        raw_df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        training_df = raw_df.loc[
            self.training_start_datetime : self.training_end_datetime
        ]
        validation_df = raw_df.loc[
            self.validation_start_datetime : self.validation_end_datetime
        ]
        test_df = raw_df.loc[self.test_start_datetime : self.test_end_datetime]

        return training_df, validation_df, test_df

    def setup(
        self,
        stage: Optional[Literal["fit", "validate", "test", "predict"]] = None,
    ):
        # Load and split data
        raw = get_raw(self.dataset_path)
        raw_df = raw.data

        training_df, validation_df, test_df = self._split_df(raw_df)
        if self.training_target_sensors is not None:
            training_df = training_df.loc[:, self.training_target_sensors]
            validation_df = validation_df.loc[:, self.training_target_sensors]
        if self.test_target_sensors is not None:
            test_df = test_df.loc[:, self.test_target_sensors]

        # Create datasets
        training_sensor_datasets = self._create_sensor_datasets(
            training_df,
            allow_nan=False,
        )
        validation_sensor_datasets = self._create_sensor_datasets(
            validation_df,
            allow_nan=False,
        )
        test_sensor_datasets = self._create_sensor_datasets(
            test_df,
            allow_nan=False,
        )

        # Concatenate datasets
        self.training_dataset = ConcatDataset(training_sensor_datasets)
        self.validation_dataset = ConcatDataset(validation_sensor_datasets)
        self.test_dataset = ConcatDataset(test_sensor_datasets)

        # Scaler fitting
        if self.scale_strictly:
            ref_data = self._get_strict_scaler_ref(self.training_dataset)
        else:
            ref_data = self._get_normal_scaler_ref(
                training_df, self.training_end_datetime
            )
        self.scaler.fit(ref_data)

        # Apply scaler to datasets
        for dataset in (
            training_sensor_datasets + validation_sensor_datasets + test_sensor_datasets
        ):
            dataset.apply_scaler(self.scaler)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.training_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle_training,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.validation_dataset,
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