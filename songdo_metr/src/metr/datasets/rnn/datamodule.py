import logging
from typing import Callable, List, Literal, Optional, Tuple

import lightning as L
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import ConcatDataset, DataLoader, Subset
from tqdm import tqdm

from metr.datasets.rnn.dataloader import collate_simple, collate_multi_sensor
from metr.datasets.rnn.dataset import TrafficDataset, TrafficDataType, TrafficMultiSensorDataset, TrafficMultiSensorDataType
from metr.components.metr_imc.traffic_data import get_raw

logger = logging.getLogger(__name__)

CollateFnInput = List[TrafficDataType]
MultiSensorCollateFnInput = List[TrafficMultiSensorDataType]


class SimpleTrafficDataModule(L.LightningDataModule):
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


class MultiSensorTrafficDataModule(L.LightningDataModule):
    """TrafficMultiSensorDataset을 사용하는 개선된 Lightning DataModule.
    
    센서별 정보를 포함한 다중 센서 데이터 처리를 위한 최적화된 DataModule.
    """
    
    def __init__(
        self,
        dataset_path: str,
        training_period: Tuple[str, str] = ("2022-11-01 00:00:00", "2024-07-31 23:59:59"),
        validation_period: Tuple[str, str] = ("2024-08-01 00:00:00", "2024-09-30 23:59:59"),
        test_period: Tuple[str, str] = ("2024-10-01 00:00:00", "2024-10-31 23:59:59"),
        seq_length: int = 24,
        batch_size: int = 64,
        num_workers: int = 1,
        shuffle_training: bool = True,
        collate_fn: Callable[[MultiSensorCollateFnInput], Tuple] = collate_multi_sensor,
        target_sensors: Optional[List[str]] = None,
        scale_method: Literal["normal", "strict"] = "normal",
    ):
        """
        Args:
            dataset_path: 데이터셋 파일 경로
            training_period: 훈련 데이터 기간 (시작, 끝)
            validation_period: 검증 데이터 기간 (시작, 끝)
            test_period: 테스트 데이터 기간 (시작, 끝)
            seq_length: 시퀀스 길이
            batch_size: 배치 크기
            num_workers: DataLoader worker 수
            shuffle_training: 훈련 데이터 셔플 여부
            collate_fn: 배치 처리 함수
            target_sensors: 대상 센서 목록 (None이면 모든 센서 사용)
            scale_method: 스케일링 방법 ("normal" 또는 "strict")
        """
        super().__init__()
        self.dataset_path = dataset_path
        self.training_period = training_period
        self.validation_period = validation_period
        self.test_period = test_period
        
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle_training = shuffle_training
        self.collate_fn = collate_fn
        self.target_sensors = target_sensors
        self.scale_method = scale_method
        
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        
        # 데이터셋 저장
        self.train_dataset: Optional[TrafficMultiSensorDataset] = None
        self.val_dataset: Optional[TrafficMultiSensorDataset] = None
        self.test_dataset: Optional[TrafficMultiSensorDataset] = None
        
    def _load_and_split_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """데이터 로드 및 기간별 분할."""
        logger.info(f"Loading data from {self.dataset_path}")
        raw = get_raw(self.dataset_path)
        raw_df = raw.data
        
        # 대상 센서 필터링
        if self.target_sensors is not None:
            logger.info(f"Filtering to target sensors: {self.target_sensors}")
            raw_df = raw_df.loc[:, self.target_sensors]
        
        # 기간별 데이터 분할
        train_df = raw_df.loc[self.training_period[0]:self.training_period[1]]
        val_df = raw_df.loc[self.validation_period[0]:self.validation_period[1]]
        test_df = raw_df.loc[self.test_period[0]:self.test_period[1]]
        
        logger.info(f"Data split - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        return train_df, val_df, test_df
    
    def _prepare_scaler(self, train_df: pd.DataFrame) -> None:
        """스케일러 준비 및 피팅."""
        if self.scale_method == "strict":
            # 훈련 데이터셋에서 실제 사용되는 데이터로만 스케일링
            temp_dataset = TrafficMultiSensorDataset(
                train_df, seq_length=self.seq_length, allow_nan=False
            )
            ref_data = self._get_strict_scaler_data(temp_dataset)
        else:
            # 전체 훈련 기간 데이터로 스케일링
            ref_data = train_df.to_numpy().reshape(-1, 1)
        
        logger.info(f"Fitting scaler with {len(ref_data)} data points using {self.scale_method} method")
        self.scaler.fit(ref_data)
    
    def _get_strict_scaler_data(self, dataset: TrafficMultiSensorDataset) -> np.ndarray:
        """Strict 방법으로 스케일러 데이터 추출."""
        data_list = []
        for i in tqdm(range(len(dataset)), desc="Extracting scaler reference data"):
            x, y, _, _, _ = dataset[i]
            data_list.append(x)
            data_list.append(y.reshape(1, 1))
        return np.concatenate(data_list, axis=0).reshape(-1, 1)
    
    def _apply_scaling(self, *datasets: TrafficMultiSensorDataset) -> None:
        """데이터셋들에 스케일링 적용."""
        logger.info("Applying scaling to datasets")
        for dataset in datasets:
            for sensor_name in dataset.sensor_names:
                sensor_dataset = dataset.sensor_datasets[sensor_name]
                sensor_dataset.apply_scaler(self.scaler)
    
    def setup(self, stage: Optional[Literal["fit", "validate", "test", "predict"]] = None):
        """데이터 준비 및 설정."""
        logger.info(f"Setting up data for stage: {stage}")
        
        # 데이터 로드 및 분할
        train_df, val_df, test_df = self._load_and_split_data()
        
        # 데이터셋 생성
        logger.info("Creating multi-sensor datasets")
        self.train_dataset = TrafficMultiSensorDataset(
            train_df, seq_length=self.seq_length, allow_nan=False
        )
        self.val_dataset = TrafficMultiSensorDataset(
            val_df, seq_length=self.seq_length, allow_nan=False
        )
        self.test_dataset = TrafficMultiSensorDataset(
            test_df, seq_length=self.seq_length, allow_nan=False
        )
        
        # 스케일러 준비 및 적용
        self._prepare_scaler(train_df)
        self._apply_scaling(self.train_dataset, self.val_dataset, self.test_dataset)
        
        logger.info(f"Setup complete - Train: {len(self.train_dataset)}, "
                   f"Val: {len(self.val_dataset)}, Test: {len(self.test_dataset)}")
    
    def train_dataloader(self) -> DataLoader:
        """훈련용 DataLoader 반환."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle_training,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )
    
    def val_dataloader(self) -> DataLoader:
        """검증용 DataLoader 반환."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )
    
    def test_dataloader(self) -> DataLoader:
        """테스트용 DataLoader 반환."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

