from pathlib import Path
from typing import Literal, Optional, Tuple

import lightning as L
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader

from metr.components.adj_mx import AdjacencyMatrix
from metr.components.metr_imc.traffic_data import TrafficData
from metr.components import MissingMasks

from .dataloader import collate_fn, collate_fn_with_missing
from .dataset import STGCNDataset, STGCNDatasetWithMissing


class STGCNDataModule(L.LightningDataModule):
    def __init__(
        self,
        dataset_dir_path: str,
        n_his: int = 12,
        n_pred: int = 3,
        batch_size: int = 64,
        num_workers: int = 1,
        shuffle_training: bool = True,
        adj_mx_filename: str = "adj_mx.pkl",
        training_data_filename: str = "metr-imc.h5",
        test_data_filename: str = "metr-imc_test.h5",
        train_val_split: float = 0.8,
    ):
        super().__init__()
        self.dataset_dir_path = Path(dataset_dir_path)
        self.adj_mx_path = self.dataset_dir_path / adj_mx_filename
        self.training_data_path = self.dataset_dir_path / training_data_filename
        self.test_data_path = self.dataset_dir_path / test_data_filename
        self.n_his = n_his
        self.n_pred = n_pred
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle_training = shuffle_training
        self.train_val_split = train_val_split

        self.adj_mx_raw: Optional[AdjacencyMatrix] = None
        self.training_dataset: Optional[STGCNDataset] = None
        self.validation_dataset: Optional[STGCNDataset] = None
        self.test_dataset: Optional[STGCNDataset] = None

        self._scaler: Optional[MinMaxScaler] = None

    @property
    def scaler(self) -> Optional[MinMaxScaler]:
        """Get the fitted scaler."""
        return self._scaler

    def _prepare_scaler(self, train_data: np.ndarray) -> None:
        """Prepare and fit the scaler on training data.

        Args:
            train_data: Training data array of shape (time_steps, n_vertex)
        """
        # Flatten data for fitting
        ref_data = train_data.reshape(-1, 1)
        ref_data = ref_data[~np.isnan(ref_data).any(axis=1)]

        self._scaler = MinMaxScaler(feature_range=(0, 1))
        self._scaler.fit(ref_data)

    def _apply_scaling(self, *datasets: STGCNDataset) -> None:
        """Apply scaling to datasets.

        Args:
            datasets: STGCNDataset instances to scale
        """
        if self._scaler is None:
            raise ValueError("Scaler is not fitted. Call _prepare_scaler first.")

        for dataset in datasets:
            dataset.apply_scaler(self._scaler)

    def setup(
        self,
        stage: Optional[Literal["fit", "validate", "test", "predict"]] = None,
    ):
        self.adj_mx_raw = AdjacencyMatrix.import_from_pickle(self.adj_mx_path)
        ordered_sensor_ids = self.adj_mx_raw.sensor_ids

        if stage in ["fit", "validate", None]:
            raw = TrafficData.import_from_hdf(self.training_data_path)
            training_data_df, validation_data_df = self._split_dataset(
                raw,
                self.train_val_split,
                ordered_sensor_ids,
            )
            training_data_array = training_data_df.values  # (time_steps, n_vertex)
            validation_data_array = validation_data_df.values  # (time_steps, n_vertex)

            self.training_dataset = STGCNDataset(
                training_data_array,
                self.n_his,
                self.n_pred,
            )
            self.validation_dataset = STGCNDataset(
                validation_data_array,
                self.n_his,
                self.n_pred,
            )

            # Warn if datasets are too small
            if len(self.training_dataset) < self.batch_size:
                print(
                    f"WARNING: Training dataset ({len(self.training_dataset)} samples) is smaller than batch_size ({self.batch_size})"
                )
            if len(self.validation_dataset) < self.batch_size:
                print(
                    f"WARNING: Validation dataset ({len(self.validation_dataset)} samples) is smaller than batch_size ({self.batch_size})"
                )

            # Prepare and apply scaling
            self._prepare_scaler(training_data_array)
            self._apply_scaling(self.training_dataset, self.validation_dataset)

        if stage in ["test", None]:
            test_data_df = self._load_test_dataset(ordered_sensor_ids)
            test_data_array = test_data_df.values  # (time_steps, n_vertex)
            self.test_dataset = STGCNDataset(test_data_array, self.n_his, self.n_pred)

            # Warn if test dataset is too small
            if len(self.test_dataset) < self.batch_size:
                print(
                    f"WARNING: Test dataset ({len(self.test_dataset)} samples) is smaller than batch_size ({self.batch_size})"
                )

            # Apply scaling to test dataset (scaler should be already fitted)
            if self._scaler is not None:
                self._apply_scaling(self.test_dataset)

    def _split_dataset_by_date(
        self, raw: TrafficData, split_date: pd.Timestamp, ordered_sensor_ids: list
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split traffic data into training and validation DataFrames by date.

        Args:
            raw: Raw traffic data to split
            split_date: Date to split the data at
            ordered_sensor_ids: List of sensor IDs to filter
            
        Returns:
            Tuple of (training_data_df, validation_data_df)
        """
        training_raw, validation_raw = raw.split(split_date)
        training_data_df = training_raw.data[ordered_sensor_ids]
        validation_data_df = validation_raw.data[ordered_sensor_ids]

        return training_data_df, validation_data_df

    def _split_dataset(
        self, raw: TrafficData, split_rate: float, ordered_sensor_ids: list
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split traffic data into training and validation DataFrames.

        Args:
            raw: Raw traffic data to split
            split_rate: Ratio for train/validation split (0.0 to 1.0)

        Returns:
            Tuple of (training_data_df, validation_data_df)
        """
        # Calculate split date based on split_rate
        total_length = len(raw.data)
        split_index = int(total_length * split_rate)
        split_date = raw.data.index[split_index]

        # Split and extract DataFrames
        return self._split_dataset_by_date(
            raw,
            split_date,
            ordered_sensor_ids,
        )

    def _load_test_dataset(self, ordered_sensor_ids: list) -> pd.DataFrame:
        """Load test data and return as filtered DataFrame.

        Returns:
            Test data DataFrame filtered by sensor IDs
        """
        test_raw = TrafficData.import_from_hdf(self.test_data_path)
        test_data_df = test_raw.data[ordered_sensor_ids]

        return test_data_df

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.training_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle_training,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.validation_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )


class STGCNDataModuleByDate(STGCNDataModule):
    def __init__(
        self,
        dataset_dir_path: str,
        n_his: int = 12,
        n_pred: int = 3,
        batch_size: int = 64,
        num_workers: int = 1,
        shuffle_training: bool = True,
        adj_mx_filename: str = "adj_mx.pkl",
        traffic_data_filename: str = "metr-imc.h5",
        training_period: Tuple[str, str] = (
            "2023-01-26 00:00:00",
            "2025-09-30 23:59:59",
        ),
        validation_period: Tuple[str, str] = (
            "2025-10-01 00:00:00",
            "2025-11-30 23:59:59",
        ),
        test_period: Tuple[str, str] = (
            "2025-12-01 00:00:00",
            "2025-12-31 23:59:59",
        ),
    ):
        super().__init__(
            dataset_dir_path,
            n_his,
            n_pred,
            batch_size,
            num_workers,
            shuffle_training,
            adj_mx_filename,
            traffic_data_filename,
        )
        self.training_period = training_period
        self.validation_period = validation_period
        self.test_period = test_period

    def _split_dataset(
        self, raw: TrafficData, split_rate: float, ordered_sensor_ids: list
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        # Todo: 추후 근본 설계 변경 필요
        """Split dataset by predefined date periods (ignores split_rate).
        
        Note: split_rate parameter is ignored as this class uses fixed date periods.
        """
        raw_df = raw.data[ordered_sensor_ids]
        train_data_df = raw_df[self.training_period[0] : self.training_period[1]]
        valid_data_df = raw_df[self.validation_period[0] : self.validation_period[1]]

        return train_data_df, valid_data_df

    def _load_test_dataset(self, ordered_sensor_ids: list) -> pd.DataFrame:
        # Todo: 추후 근본 설계 변경 필요
        """Load test dataset from predefined date period.
        
        Note: Reloads data from file to avoid caching issues with parent class.
        """
        raw = TrafficData.import_from_hdf(self.training_data_path)
        raw_df = raw.data[ordered_sensor_ids]
        test_data_df = raw_df[self.test_period[0] : self.test_period[1]]

        return test_data_df


class STGCNSplitDataModule(L.LightningDataModule):
    """STGCN DataModule with separate training and test dataset files.
    
    Training 데이터와 Test 데이터를 별도 파일에서 로드하며,
    Training 데이터는 비율에 따라 train/validation으로 분할합니다.
    Test 데이터는 missing mask 정보를 포함하여 보간된 데이터 구분이 가능합니다.
    """

    def __init__(
        self,
        training_data_path: str,
        test_data_path: str,
        test_missing_path: str,
        adj_mx_path: str,
        n_his: int = 12,
        n_pred: int = 3,
        batch_size: int = 64,
        num_workers: int = 1,
        shuffle_training: bool = True,
        train_val_split: float = 0.8,
    ):
        """
        Args:
            dataset_dir_path: 데이터셋 디렉토리 경로
            training_data_path: 학습용 데이터 파일 경로 (.h5)
            test_data_path: 테스트용 데이터 파일 경로 (.h5)
            test_missing_path: 테스트 데이터의 missing mask 파일 경로 (.h5)
            adj_mx_path: Adjacency matrix 파일 경로 (.pkl)
            n_his: Historical time steps (입력 시퀀스 길이)
            n_pred: Prediction horizon (예측 시점)
            batch_size: 배치 크기
            num_workers: DataLoader worker 수
            shuffle_training: 훈련 데이터 셔플 여부
            train_val_split: Train/Validation 분할 비율 (기본값: 0.8)
        """
        super().__init__()
        self.training_data_path = Path(training_data_path)
        self.test_data_path = Path(test_data_path)
        self.test_missing_path = Path(test_missing_path)
        self.adj_mx_path = Path(adj_mx_path)

        self.n_his = n_his
        self.n_pred = n_pred
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle_training = shuffle_training
        self.train_val_split = train_val_split

        self.adj_mx_raw: Optional[AdjacencyMatrix] = None
        self.training_dataset: Optional[STGCNDataset] = None
        self.validation_dataset: Optional[STGCNDataset] = None
        self.test_dataset: Optional[STGCNDatasetWithMissing] = None

        self._scaler: Optional[MinMaxScaler] = None

    @property
    def scaler(self) -> Optional[MinMaxScaler]:
        """Get the fitted scaler."""
        return self._scaler

    def _prepare_scaler(self, train_data: np.ndarray) -> None:
        """Prepare and fit the scaler on training data."""
        ref_data = train_data.reshape(-1, 1)
        ref_data = ref_data[~np.isnan(ref_data).any(axis=1)]

        self._scaler = MinMaxScaler(feature_range=(0, 1))
        self._scaler.fit(ref_data)

    def _apply_scaling(self, *datasets) -> None:
        """Apply scaling to datasets."""
        if self._scaler is None:
            raise ValueError("Scaler is not fitted. Call _prepare_scaler first.")

        for dataset in datasets:
            dataset.apply_scaler(self._scaler)

    def _load_training_data(
        self, ordered_sensor_ids: list
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """학습 데이터 로드 및 train/val 분할."""
        raw = TrafficData.import_from_hdf(self.training_data_path)
        raw_df = raw.data[ordered_sensor_ids]

        # 시간순 분할 (비율 기반)
        total_rows = len(raw_df)
        split_idx = int(total_rows * self.train_val_split)

        train_df = raw_df.iloc[:split_idx]
        val_df = raw_df.iloc[split_idx:]

        print(
            f"Training data split - Train: {len(train_df)} rows "
            f"({self.train_val_split * 100:.0f}%), "
            f"Val: {len(val_df)} rows ({(1 - self.train_val_split) * 100:.0f}%)"
        )

        return train_df, val_df

    def _load_test_data(
        self, ordered_sensor_ids: list
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """테스트 데이터 및 missing mask 로드."""
        # 테스트 데이터 로드
        raw = TrafficData.import_from_hdf(self.test_data_path)
        raw_df = raw.data[ordered_sensor_ids]

        # Missing mask 로드
        missing_masks = MissingMasks.import_from_hdf(self.test_missing_path)
        missing_mask_df = missing_masks.data[ordered_sensor_ids]

        # DataFrame과 mask의 인덱스 정렬
        missing_mask_aligned = missing_mask_df.reindex(
            index=raw_df.index, columns=raw_df.columns, fill_value=False
        )

        print(f"Test data loaded: {len(raw_df)} rows")

        return raw_df, missing_mask_aligned.values

    def setup(
        self,
        stage: Optional[Literal["fit", "validate", "test", "predict"]] = None,
    ):
        # Adjacency matrix 로드 및 센서 ID 순서 획득
        self.adj_mx_raw = AdjacencyMatrix.import_from_pickle(self.adj_mx_path)
        ordered_sensor_ids = self.adj_mx_raw.sensor_ids

        if stage in ["fit", "validate", None]:
            # 학습 데이터 로드 및 분할
            train_df, val_df = self._load_training_data(ordered_sensor_ids)

            training_data_array = train_df.values
            validation_data_array = val_df.values

            # 데이터셋 생성 (Train/Val은 기본 STGCNDataset 사용)
            self.training_dataset = STGCNDataset(
                training_data_array, self.n_his, self.n_pred
            )
            self.validation_dataset = STGCNDataset(
                validation_data_array, self.n_his, self.n_pred
            )

            # 경고 출력
            if len(self.training_dataset) < self.batch_size:
                print(
                    f"WARNING: Training dataset ({len(self.training_dataset)} samples) "
                    f"is smaller than batch_size ({self.batch_size})"
                )
            if len(self.validation_dataset) < self.batch_size:
                print(
                    f"WARNING: Validation dataset ({len(self.validation_dataset)} samples) "
                    f"is smaller than batch_size ({self.batch_size})"
                )

            # Scaler 준비 및 적용
            self._prepare_scaler(training_data_array)
            self._apply_scaling(self.training_dataset, self.validation_dataset)

        if stage in ["test", None]:
            # 테스트 데이터 및 missing mask 로드
            test_df, test_missing_mask = self._load_test_data(ordered_sensor_ids)
            test_data_array = test_df.values

            # 테스트 데이터셋 생성 (missing mask 포함)
            self.test_dataset = STGCNDatasetWithMissing(
                test_data_array, self.n_his, self.n_pred, missing_mask=test_missing_mask
            )

            if len(self.test_dataset) < self.batch_size:
                print(
                    f"WARNING: Test dataset ({len(self.test_dataset)} samples) "
                    f"is smaller than batch_size ({self.batch_size})"
                )

            # Scaler 적용 (이미 fit된 경우)
            if self._scaler is not None:
                self._apply_scaling(self.test_dataset)

    def train_dataloader(self) -> DataLoader:
        """훈련용 DataLoader 반환."""
        return DataLoader(
            self.training_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle_training,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        """검증용 DataLoader 반환."""
        return DataLoader(
            self.validation_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        """테스트용 DataLoader 반환 (missing 정보 포함)."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn_with_missing,
        )