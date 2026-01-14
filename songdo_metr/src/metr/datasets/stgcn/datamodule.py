from pathlib import Path
from typing import Literal, Optional, Tuple

import lightning as L
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader

from metr.components.adj_mx import AdjacencyMatrix
from metr.components.metr_imc.traffic_data import TrafficData

from .dataloader import collate_fn
from .dataset import STGCNDataset


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
