"""
MLCAFormer DataModule for PyTorch Lightning.
"""
import logging
from pathlib import Path
from typing import Callable, List, Literal, Optional, Tuple

import lightning as L
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from metr.components.metr_imc.traffic_data import TrafficData, get_raw

from .dataloader import collate_fn
from .dataset import MLCAFormerDataset

logger = logging.getLogger(__name__)


class MLCAFormerDataModule(L.LightningDataModule):
    """PyTorch Lightning DataModule for MLCAFormer.
    
    This DataModule handles data loading, splitting, scaling, and DataLoader
    creation for the MLCAFormer model.
    
    Args:
        dataset_path: Path to the HDF5 dataset file
        training_period: Tuple of (start_datetime, end_datetime) for training
        validation_period: Tuple of (start_datetime, end_datetime) for validation
        test_period: Tuple of (start_datetime, end_datetime) for testing
        in_steps: Number of input time steps (default: 12)
        out_steps: Number of output time steps (default: 12)
        steps_per_day: Number of time steps per day (default: 288)
        batch_size: Batch size for DataLoader (default: 64)
        num_workers: Number of DataLoader workers (default: 0)
        shuffle_training: Whether to shuffle training data (default: True)
        collate_fn: Collate function for DataLoader (default: collate_fn)
        target_sensors: List of sensor IDs to use (default: None = all sensors)
        scale_method: Scaling method - "normal", "strict", or "none" (default: "normal")
    """
    
    def __init__(
        self,
        dataset_path: str,
        training_period: Tuple[str, str] = ("2022-11-01 00:00:00", "2024-07-31 23:59:59"),
        validation_period: Tuple[str, str] = ("2024-08-01 00:00:00", "2024-09-30 23:59:59"),
        test_period: Tuple[str, str] = ("2024-10-01 00:00:00", "2024-10-31 23:59:59"),
        in_steps: int = 12,
        out_steps: int = 12,
        steps_per_day: int = 288,
        batch_size: int = 64,
        num_workers: int = 0,
        shuffle_training: bool = True,
        collate_fn: Callable = collate_fn,
        target_sensors: Optional[List[str]] = None,
        scale_method: Optional[Literal["normal", "strict", "none"]] = "normal",
    ):
        super().__init__()
        self.dataset_path = dataset_path
        self.training_period = training_period
        self.validation_period = validation_period
        self.test_period = test_period
        
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.steps_per_day = steps_per_day
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle_training = shuffle_training
        self.collate_fn = collate_fn
        self.target_sensors = target_sensors
        self.scale_method = scale_method
        
        self._scaler: Optional[MinMaxScaler] = None
        
        # Dataset placeholders
        self.train_dataset: Optional[MLCAFormerDataset] = None
        self.val_dataset: Optional[MLCAFormerDataset] = None
        self.test_dataset: Optional[MLCAFormerDataset] = None
        
        # Metadata
        self.num_nodes: Optional[int] = None
        self.sensor_ids: Optional[List[str]] = None
    
    @property
    def scaler(self) -> Optional[MinMaxScaler]:
        """Get the fitted scaler. Creates one if not exists."""
        if self.scale_method is None or self.scale_method == "none":
            return None
        
        if self._scaler is None:
            logger.info("Scaler not found. Creating scaler from training data...")
            train_df, _, _ = self._load_and_split_data()
            self._prepare_scaler(train_df)
        
        return self._scaler
    
    def _load_and_split_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load data and split by periods.
        
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        logger.info(f"Loading data from {self.dataset_path}")
        raw = get_raw(self.dataset_path)
        raw_df = raw.data
        
        # Filter target sensors if specified
        if self.target_sensors is not None:
            logger.info(f"Filtering to {len(self.target_sensors)} target sensors")
            raw_df = raw_df.loc[:, self.target_sensors]
        
        # Split by periods
        train_df = raw_df.loc[self.training_period[0]:self.training_period[1]]
        val_df = raw_df.loc[self.validation_period[0]:self.validation_period[1]]
        test_df = raw_df.loc[self.test_period[0]:self.test_period[1]]
        
        logger.info(f"Data split - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        return train_df, val_df, test_df
    
    def _prepare_scaler(self, train_df: pd.DataFrame) -> None:
        """Prepare and fit the scaler on training data.
        
        Args:
            train_df: Training DataFrame
        """
        if self.scale_method is None or self.scale_method == "none":
            logger.info("Skipping scaler preparation (scale_method is None or 'none')")
            self._scaler = None
            return
        
        if self.scale_method == "strict":
            # Use only actual training samples for fitting
            temp_dataset = MLCAFormerDataset(
                train_df,
                in_steps=self.in_steps,
                out_steps=self.out_steps,
                steps_per_day=self.steps_per_day,
            )
            ref_data = self._get_strict_scaler_data(temp_dataset)
        else:
            # Use all training period data (excluding NaN)
            ref_data = train_df.values.reshape(-1, 1)
            ref_data = ref_data[~np.isnan(ref_data).any(axis=1)]
        
        logger.info(f"Fitting scaler with {len(ref_data)} data points using '{self.scale_method}' method")
        self._scaler = MinMaxScaler(feature_range=(0, 1))
        self._scaler.fit(ref_data)
    
    def _get_strict_scaler_data(self, dataset: MLCAFormerDataset) -> np.ndarray:
        """Extract data for strict scaling method.
        
        Args:
            dataset: MLCAFormerDataset instance
            
        Returns:
            Array of shape (n_values, 1) for scaler fitting
        """
        data_list = []
        for i in tqdm(range(len(dataset)), desc="Extracting scaler reference data"):
            x, y = dataset[i]
            # Only traffic values (first channel of x, all of y)
            data_list.append(x[:, :, 0].numpy().flatten())
            data_list.append(y[:, :, 0].numpy().flatten())
        
        return np.concatenate(data_list).reshape(-1, 1)
    
    def _apply_scaling(self, *datasets: MLCAFormerDataset) -> None:
        """Apply scaling to datasets.
        
        Args:
            datasets: MLCAFormerDataset instances to scale
        """
        if self.scale_method is None or self.scale_method == "none":
            logger.info("Skipping scaling (scale_method is None or 'none')")
            return
        
        if self._scaler is None:
            raise ValueError("Scaler is not fitted. Call _prepare_scaler first.")
        
        logger.info(f"Applying scaling to {len(datasets)} datasets")
        for dataset in datasets:
            dataset.apply_scaler(self._scaler)
    
    def setup(self, stage: Optional[Literal["fit", "validate", "test", "predict"]] = None) -> None:
        """Setup datasets for training/validation/testing.
        
        Args:
            stage: Current stage (fit, validate, test, predict)
        """
        logger.info(f"Setting up MLCAFormer data for stage: {stage}")
        
        # Load and split data
        train_df, val_df, test_df = self._load_and_split_data()
        
        # Store metadata
        self.num_nodes = train_df.shape[1]
        self.sensor_ids = list(train_df.columns)
        
        logger.info(f"Number of nodes: {self.num_nodes}")
        
        # Create datasets
        logger.info("Creating MLCAFormer datasets...")
        
        self.train_dataset = MLCAFormerDataset(
            train_df,
            in_steps=self.in_steps,
            out_steps=self.out_steps,
            steps_per_day=self.steps_per_day,
        )
        
        self.val_dataset = MLCAFormerDataset(
            val_df,
            in_steps=self.in_steps,
            out_steps=self.out_steps,
            steps_per_day=self.steps_per_day,
        )
        
        self.test_dataset = MLCAFormerDataset(
            test_df,
            in_steps=self.in_steps,
            out_steps=self.out_steps,
            steps_per_day=self.steps_per_day,
        )
        
        # Prepare and apply scaling
        self._prepare_scaler(train_df)
        self._apply_scaling(self.train_dataset, self.val_dataset, self.test_dataset)
        
        logger.info(f"Setup complete - Train: {len(self.train_dataset)}, "
                   f"Val: {len(self.val_dataset)}, Test: {len(self.test_dataset)}")
    
    def train_dataloader(self) -> DataLoader:
        """Create training DataLoader."""
        if self.train_dataset is None:
            raise RuntimeError("train_dataset is None. Call setup() first.")
        
        kwargs = {
            "batch_size": self.batch_size,
            "shuffle": self.shuffle_training,
            "collate_fn": self.collate_fn,
        }
        
        if self.num_workers > 0:
            kwargs["num_workers"] = self.num_workers
            kwargs["persistent_workers"] = True
        
        return DataLoader(self.train_dataset, **kwargs)
    
    def val_dataloader(self) -> DataLoader:
        """Create validation DataLoader."""
        if self.val_dataset is None:
            raise RuntimeError("val_dataset is None. Call setup() first.")
        
        kwargs = {
            "batch_size": self.batch_size,
            "shuffle": False,
            "collate_fn": self.collate_fn,
        }
        
        if self.num_workers > 0:
            kwargs["num_workers"] = self.num_workers
            kwargs["persistent_workers"] = True
        
        return DataLoader(self.val_dataset, **kwargs)
    
    def test_dataloader(self) -> DataLoader:
        """Create test DataLoader."""
        if self.test_dataset is None:
            raise RuntimeError("test_dataset is None. Call setup() first.")
        
        kwargs = {
            "batch_size": self.batch_size,
            "shuffle": False,
            "collate_fn": self.collate_fn,
        }
        
        if self.num_workers > 0:
            kwargs["num_workers"] = self.num_workers
            kwargs["persistent_workers"] = True
        
        return DataLoader(self.test_dataset, **kwargs)
