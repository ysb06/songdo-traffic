from pathlib import Path
from typing import Literal, Optional

import lightning as L
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from metr.components.adj_mx import AdjacencyMatrix
from metr.components.metr_imc.traffic_data import TrafficData

from .dataloader import collate_fn
from .dataset import DCRNNDataset


class DCRNNDataModule(L.LightningDataModule):
    """Lightning DataModule for DCRNN model.

    Handles data loading, preprocessing, and batching for DCRNN training.
    Uses StandardScaler for normalization (DCRNN standard).

    Args:
        dataset_dir_path: Path to dataset directory
        seq_len: Number of historical time steps (default: 12)
        horizon: Number of prediction time steps (default: 12)
        batch_size: Batch size for training (default: 64)
        num_workers: Number of DataLoader workers (default: 1)
        shuffle_training: Whether to shuffle training data (default: True)
        adj_mx_filename: Adjacency matrix pickle filename
        training_data_filename: Training data HDF5 filename
        test_data_filename: Test data HDF5 filename
        train_val_split: Train/validation split ratio (default: 0.8)
        add_time_in_day: Whether to add time-of-day feature (default: True)
        add_day_in_week: Whether to add day-of-week feature (default: False)
    """

    def __init__(
        self,
        dataset_dir_path: str,
        seq_len: int = 12,
        horizon: int = 12,
        batch_size: int = 64,
        num_workers: int = 1,
        shuffle_training: bool = True,
        adj_mx_filename: str = "adj_mx.pkl",
        training_data_filename: str = "metr-imc_train.h5",
        test_data_filename: str = "metr-imc_test.h5",
        train_val_split: float = 0.8,
        add_time_in_day: bool = True,
        add_day_in_week: bool = False,
    ):
        super().__init__()
        self.dataset_dir_path = Path(dataset_dir_path)
        self.adj_mx_path = self.dataset_dir_path / adj_mx_filename
        self.training_data_path = self.dataset_dir_path / training_data_filename
        self.test_data_path = self.dataset_dir_path / test_data_filename

        self.seq_len = seq_len
        self.horizon = horizon
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle_training = shuffle_training
        self.train_val_split = train_val_split
        self.add_time_in_day = add_time_in_day
        self.add_day_in_week = add_day_in_week

        # Will be set during setup
        self.adj_mx_raw: Optional[AdjacencyMatrix] = None
        self.scaler: Optional[StandardScaler] = None
        self.training_dataset: Optional[DCRNNDataset] = None
        self.validation_dataset: Optional[DCRNNDataset] = None
        self.test_dataset: Optional[DCRNNDataset] = None

    @property
    def adj_mx(self) -> np.ndarray:
        """Return the adjacency matrix as numpy array."""
        if self.adj_mx_raw is None:
            raise ValueError("DataModule not setup. Call setup() first.")
        return self.adj_mx_raw.adj_mx

    @property
    def num_nodes(self) -> int:
        """Return the number of nodes (sensors)."""
        if self.adj_mx_raw is None:
            raise ValueError("DataModule not setup. Call setup() first.")
        return len(self.adj_mx_raw.sensor_ids)

    @property
    def input_dim(self) -> int:
        """Return input dimension (1 + time features)."""
        dim = 1
        if self.add_time_in_day:
            dim += 1
        if self.add_day_in_week:
            dim += 7
        return dim

    @property
    def output_dim(self) -> int:
        """Return output dimension (traffic value only)."""
        return 1

    def setup(
        self,
        stage: Optional[Literal["fit", "validate", "test", "predict"]] = None,
    ):
        """Setup datasets and scaler.

        Args:
            stage: Current stage ('fit', 'validate', 'test', or 'predict')
        """
        # Load adjacency matrix
        self.adj_mx_raw = AdjacencyMatrix.import_from_pickle(str(self.adj_mx_path))
        ordered_sensor_ids = self.adj_mx_raw.sensor_ids

        if stage in ["fit", "validate", None]:
            # Load training data
            raw = TrafficData.import_from_hdf(str(self.training_data_path))

            # Calculate split date based on train_val_split ratio
            total_length = len(raw.data)
            split_index = int(total_length * self.train_val_split)
            split_date = raw.data.index[split_index]

            training_raw, validation_raw = raw.split(split_date)

            # Reorder columns to match adjacency matrix sensor order
            training_df = training_raw.data[ordered_sensor_ids]
            validation_df = validation_raw.data[ordered_sensor_ids]

            # Fit scaler on training data (traffic values only)
            self._fit_scaler(training_df)

            # Apply scaling to traffic data
            training_df_scaled = self._apply_scaling(training_df)
            validation_df_scaled = self._apply_scaling(validation_df)

            # Create datasets
            self.training_dataset = DCRNNDataset(
                training_df_scaled,
                seq_len=self.seq_len,
                horizon=self.horizon,
                add_time_in_day=self.add_time_in_day,
                add_day_in_week=self.add_day_in_week,
            )
            self.validation_dataset = DCRNNDataset(
                validation_df_scaled,
                seq_len=self.seq_len,
                horizon=self.horizon,
                add_time_in_day=self.add_time_in_day,
                add_day_in_week=self.add_day_in_week,
            )

        if stage in ["test", None]:
            # Load test data
            test_raw = TrafficData.import_from_hdf(str(self.test_data_path))
            test_df = test_raw.data[ordered_sensor_ids]

            # Apply scaling (scaler should already be fitted)
            if self.scaler is None:
                # If test-only, fit scaler on training data
                raw = TrafficData.import_from_hdf(str(self.training_data_path))
                training_df = raw.data[ordered_sensor_ids]
                self._fit_scaler(training_df)

            test_df_scaled = self._apply_scaling(test_df)

            self.test_dataset = DCRNNDataset(
                test_df_scaled,
                seq_len=self.seq_len,
                horizon=self.horizon,
                add_time_in_day=self.add_time_in_day,
                add_day_in_week=self.add_day_in_week,
            )

    def _fit_scaler(self, df) -> None:
        """Fit StandardScaler on training data.

        Args:
            df: Training DataFrame
        """
        # Flatten all traffic values for fitting
        data = df.values.flatten().reshape(-1, 1)
        # Remove NaN values
        data = data[~np.isnan(data).flatten()]
        data = data.reshape(-1, 1)

        self.scaler = StandardScaler()
        self.scaler.fit(data)

    def _apply_scaling(self, df):
        """Apply StandardScaler to DataFrame.

        Args:
            df: DataFrame to scale

        Returns:
            Scaled DataFrame with same index and columns
        """
        import pandas as pd

        assert self.scaler is not None, "Scaler must be fitted before applying scaling"
        scaled_values = self.scaler.transform(df.values.reshape(-1, 1))
        scaled_values = scaled_values.reshape(df.shape)
        return pd.DataFrame(scaled_values, index=df.index, columns=df.columns)

    def train_dataloader(self) -> DataLoader:
        """Return training DataLoader."""
        assert self.training_dataset is not None, "Call setup() first"
        return DataLoader(
            self.training_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle_training,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        """Return validation DataLoader."""
        assert self.validation_dataset is not None, "Call setup() first"
        return DataLoader(
            self.validation_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        """Return test DataLoader."""
        assert self.test_dataset is not None, "Call setup() first"
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )
