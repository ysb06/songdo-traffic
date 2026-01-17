import logging
from typing import Callable, List, Literal, Optional, Tuple

import lightning as L
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import ConcatDataset, DataLoader, Subset
from tqdm import tqdm

from metr.datasets.rnn.dataloader import collate_simple, collate_multi_sensor, collate_multi_sensor_with_missing
from metr.datasets.rnn.dataset import (
    TrafficDataset,
    TrafficDataType,
    TrafficMultiSensorDataset,
    TrafficMultiSensorDataType,
    TrafficMultiSensorWithMissingDataType,
)
from metr.components.metr_imc.traffic_data import get_raw
from metr.components import MissingMasks

logger = logging.getLogger(__name__)

CollateFnInput = List[TrafficDataType]
MultiSensorCollateFnInput = List[TrafficMultiSensorDataType]
MultiSensorWithMissingCollateFnInput = List[TrafficMultiSensorWithMissingDataType]


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
        self._scaler = None

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
        # NaN 값을 제거하여 스케일러가 올바르게 피팅되도록 함
        ref_data = scaling_target.to_numpy().reshape(-1, 1)
        ref_data = ref_data[~np.isnan(ref_data).any(axis=1)]
        return ref_data

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
        self._scaler = MinMaxScaler(feature_range=(0, 1))
        self._scaler.fit(ref_data)

        # Apply scaler to datasets
        for dataset in (
            training_sensor_datasets + validation_sensor_datasets + test_sensor_datasets
        ):
            dataset.apply_scaler(self._scaler)

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

    @property
    def scaler(self):
        """스케일러를 반환합니다. None인 경우 자동으로 생성합니다."""
        if self._scaler is None:
            logger.info("Scaler not found. Creating scaler from training data...")
            self._create_scaler()
        return self._scaler
    
    def _create_scaler(self) -> None:
        """스케일러가 None인 경우 훈련 데이터로부터 스케일러를 생성합니다."""
        # 데이터 로드 및 분할
        raw = get_raw(self.dataset_path)
        raw_df = raw.data
        training_df, _, _ = self._split_df(raw_df)
        
        # 대상 센서 필터링
        if self.training_target_sensors is not None:
            training_df = training_df.loc[:, self.training_target_sensors]
        
        # 스케일러 준비
        if self.scale_strictly:
            # 엄격한 스케일링을 위해 임시 데이터셋 생성
            temp_datasets = self._create_sensor_datasets(training_df, allow_nan=False)
            temp_concat_dataset = ConcatDataset(temp_datasets)
            ref_data = self._get_strict_scaler_ref(temp_concat_dataset)
        else:
            ref_data = self._get_normal_scaler_ref(training_df, self.training_end_datetime)
        
        self._scaler = MinMaxScaler(feature_range=(0, 1))
        self._scaler.fit(ref_data)

# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #

class MultiSensorTrafficDataModule(L.LightningDataModule):
    """TrafficMultiSensorDataset을 사용하는 개선된 Lightning DataModule.
    
    센서별 정보를 포함한 다중 센서 데이터 처리를 위한 최적화된 DataModule.
    Training 데이터와 Test 데이터를 별도 파일에서 로드하며,
    Training 데이터는 비율에 따라 train/validation으로 분할합니다.
    """
    
    def __init__(
        self,
        training_dataset_path: str,
        test_dataset_path: str,
        test_missing_path: str,
        train_val_split: float = 0.8,
        seq_length: int = 24,
        batch_size: int = 512,
        num_workers: int = 0,
        shuffle_training: bool = True,
        collate_fn: Callable[[MultiSensorCollateFnInput], Tuple] = collate_multi_sensor,
        test_collate_fn: Callable[[MultiSensorWithMissingCollateFnInput], Tuple] = collate_multi_sensor_with_missing,
        target_sensors: Optional[List[str]] = None,
        scale_method: Optional[Literal["normal", "strict", "none"]] = "normal",
    ):
        """
        Args:
            training_dataset_path: 학습용 데이터셋 파일 경로 (.h5)
            test_dataset_path: 테스트용 데이터셋 파일 경로 (.h5)
            test_missing_path: 테스트 데이터의 missing mask 파일 경로 (.h5)
            train_val_split: Train/Validation 분할 비율 (기본값: 0.8 = 80% train, 20% val)
            seq_length: 시퀀스 길이
            batch_size: 배치 크기
            num_workers: DataLoader worker 수
            shuffle_training: 훈련 데이터 셔플 여부
            collate_fn: Train/Validation용 배치 처리 함수
            test_collate_fn: Test용 배치 처리 함수 (missing 정보 포함)
            target_sensors: 대상 센서 목록 (None이면 모든 센서 사용)
            scale_method: 스케일링 방법 ("normal", "strict", "none" 또는 None)
                - "normal": 전체 훈련 기간 데이터로 스케일링
                - "strict": 실제 사용되는 데이터로만 스케일링
                - "none" 또는 None: 스케일링 적용 안함
        """
        super().__init__()
        self.training_dataset_path = training_dataset_path
        self.test_dataset_path = test_dataset_path
        self.test_missing_path = test_missing_path
        self.train_val_split = train_val_split
        
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle_training = shuffle_training
        self.collate_fn = collate_fn
        self.test_collate_fn = test_collate_fn
        self.target_sensors = target_sensors
        self.scale_method = scale_method
        
        self._scaler = None
        
        # 데이터셋 저장
        self.train_dataset: Optional[TrafficMultiSensorDataset] = None
        self.val_dataset: Optional[TrafficMultiSensorDataset] = None
        self.test_dataset: Optional[TrafficMultiSensorDataset] = None

    @property
    def scaler(self):
        # 스케일링 적용하지 않는 경우
        if self.scale_method is None or self.scale_method == "none":
            return None
            
        if self._scaler is None:
            logger.info("Scaler not found. Creating scaler from training data...")
            train_df, _ = self._load_training_data()
            self._prepare_scaler(train_df)

        return self._scaler

    def _load_training_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """학습 데이터 로드 및 train/val 분할."""
        logger.info(f"Loading training data from {self.training_dataset_path}")
        raw = get_raw(self.training_dataset_path)
        raw_df = raw.data
        
        # 대상 센서 필터링
        if self.target_sensors is not None:
            logger.info(f"Filtering to target sensors: {len(self.target_sensors)} sensors")
            raw_df = raw_df.loc[:, self.target_sensors]
        
        # 시간순 분할 (비율 기반)
        total_rows = len(raw_df)
        split_idx = int(total_rows * self.train_val_split)
        
        train_df = raw_df.iloc[:split_idx]
        val_df = raw_df.iloc[split_idx:]
        
        logger.info(
            f"Training data split - Train: {len(train_df)} rows "
            f"({self.train_val_split * 100:.0f}%), "
            f"Val: {len(val_df)} rows ({(1 - self.train_val_split) * 100:.0f}%)"
        )
        
        return train_df, val_df

    def _load_test_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """테스트 데이터 및 missing mask 로드."""
        logger.info(f"Loading test data from {self.test_dataset_path}")
        raw = get_raw(self.test_dataset_path)
        raw_df = raw.data
        
        # Missing mask 로드
        logger.info(f"Loading test missing mask from {self.test_missing_path}")
        missing_masks = MissingMasks.import_from_hdf(self.test_missing_path)
        missing_mask = missing_masks.data
        
        # 대상 센서 필터링
        if self.target_sensors is not None:
            raw_df = raw_df.loc[:, self.target_sensors]
            missing_mask = missing_mask.loc[:, self.target_sensors]
        
        logger.info(f"Test data loaded: {len(raw_df)} rows")
        
        return raw_df, missing_mask
    
    def _prepare_scaler(self, train_df: pd.DataFrame) -> None:
        """스케일러 준비 및 피팅."""
        # 스케일링 적용하지 않는 경우
        if self.scale_method is None or self.scale_method == "none":
            logger.info("Skipping scaler preparation (scale_method is None or 'none')")
            self._scaler = None
            return
            
        elif self.scale_method == "strict":
            # 훈련 데이터셋에서 실제 사용되는 데이터로만 스케일링
            temp_dataset = TrafficMultiSensorDataset(
                train_df, seq_length=self.seq_length, allow_nan=False
            )
            ref_data = self._get_strict_scaler_data(temp_dataset)
        else:
            # 전체 훈련 기간 데이터로 스케일링 (NaN 값 제외)
            # NaN 값을 제거하여 스케일러가 올바르게 피팅되도록 함
            ref_data = train_df.to_numpy().reshape(-1, 1)
            # NaN 값이 있는 행을 제거
            ref_data = ref_data[~np.isnan(ref_data).any(axis=1)]
        
        logger.info(f"Fitting scaler with {len(ref_data)} data points using {self.scale_method} method")
        self._scaler = MinMaxScaler(feature_range=(0, 1))
        self._scaler.fit(ref_data)
    
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
        # 스케일링 적용하지 않는 경우
        if self.scale_method is None or self.scale_method == "none":
            logger.warning("Skipping scaling application (scale_method is None or 'none')")
            return
            
        logger.info("Applying scaling to datasets")
        for dataset in datasets:
            for sensor_name in dataset.sensor_names:
                sensor_dataset = dataset.sensor_datasets[sensor_name]
                sensor_dataset.apply_scaler(self._scaler)
    
    def setup(self, stage: Optional[Literal["fit", "validate", "test", "predict"]] = None):
        """데이터 준비 및 설정."""
        logger.info(f"Setting up data for stage: {stage}")
        
        # 학습 데이터 로드 및 분할
        train_df, val_df = self._load_training_data()
        
        # 테스트 데이터 및 missing mask 로드
        test_df, test_missing_mask = self._load_test_data()
        
        # 데이터셋 생성
        logger.info("Creating multi-sensor datasets")
        self.train_dataset = TrafficMultiSensorDataset(
            train_df, seq_length=self.seq_length, allow_nan=False
        )
        self.val_dataset = TrafficMultiSensorDataset(
            val_df, seq_length=self.seq_length, allow_nan=False
        )
        # 테스트 데이터셋은 missing_mask 포함
        self.test_dataset = TrafficMultiSensorDataset(
            test_df, seq_length=self.seq_length, allow_nan=False,
            missing_mask=test_missing_mask
        )
        
        # 스케일러 준비 및 적용 (train_df 기준)
        self._prepare_scaler(train_df)
        self._apply_scaling(self.train_dataset, self.val_dataset, self.test_dataset)
        
        logger.info(f"Setup complete - Train: {len(self.train_dataset)}, "
                   f"Val: {len(self.val_dataset)}, Test: {len(self.test_dataset)}")
    
    def train_dataloader(self) -> DataLoader:
        """훈련용 DataLoader 반환."""
        if self.train_dataset is None:
            raise ValueError("Train dataset is not initialized. Call setup() first.")
        
        if self.num_workers > 0:
            return DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=self.shuffle_training,
                num_workers=self.num_workers,
                persistent_workers=True,
                collate_fn=self.collate_fn,
            )
        else:
            return DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=self.shuffle_training,
                collate_fn=self.collate_fn,
            )
    
    def val_dataloader(self) -> DataLoader:
        """검증용 DataLoader 반환."""
        if self.val_dataset is None:
            raise ValueError("Validation dataset is not initialized. Call setup() first.")
        
        if self.num_workers > 0:
            return DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                persistent_workers=True,
                collate_fn=self.collate_fn,
            )
        else:
            return DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=self.collate_fn,
            )
    
    def test_dataloader(self) -> DataLoader:
        """테스트용 DataLoader 반환. missing 정보 포함 가능."""
        if self.test_dataset is None:
            raise ValueError("Test dataset is not initialized. Call setup() first.")
        
        if self.num_workers > 0:
            return DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                persistent_workers=True,
                collate_fn=self.test_collate_fn,
            )
        else:
            return DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=self.test_collate_fn,
            )
