"""
AGCRN DataModule for PyTorch Lightning.

Inherits from MLCAFormerDataModule and overrides dataset creation
to use AGCRNDataset which outputs only traffic values (no ToD/DoW features).
"""
import logging
from typing import Callable, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from metr.datasets.mlcaformer.datamodule import MLCAFormerDataModule

from .dataloader import collate_fn
from .dataset import AGCRNDataset

logger = logging.getLogger(__name__)


class AGCRNDataModule(MLCAFormerDataModule):
    """PyTorch Lightning DataModule for AGCRN.
    
    This DataModule inherits from MLCAFormerDataModule but creates AGCRNDataset
    instances which output only traffic values without temporal features.
    
    AGCRN learns spatial relationships through adaptive graph convolution
    using learnable node embeddings, so it doesn't require ToD/DoW features.
    
    Args:
        dataset_path: Path to the HDF5 dataset file
        training_period: Tuple of (start_datetime, end_datetime) for training
        validation_period: Tuple of (start_datetime, end_datetime) for validation
        test_period: Tuple of (start_datetime, end_datetime) for testing
        in_steps: Number of input time steps (default: 12)
        out_steps: Number of output time steps (default: 12)
        batch_size: Batch size for DataLoader (default: 64)
        num_workers: Number of DataLoader workers (default: 0)
        shuffle_training: Whether to shuffle training data (default: True)
        collate_fn: Collate function for DataLoader (default: agcrn collate_fn)
        target_sensors: List of sensor IDs to use (default: None = all sensors)
        scale_method: Scaling method - "normal", "strict", or "none" (default: "normal")
        normalizer: Normalizer type - "minmax" or "std" (default: "std")
    
    Data Shapes:
        Input (x): (batch_size, in_steps, n_vertex, 1)
        Target (y): (batch_size, out_steps, n_vertex, 1)
    """
    
    def __init__(
        self,
        dataset_path: str,
        training_period: Tuple[str, str] = ("2022-11-01 00:00:00", "2024-07-31 23:59:59"),
        validation_period: Tuple[str, str] = ("2024-08-01 00:00:00", "2024-09-30 23:59:59"),
        test_period: Tuple[str, str] = ("2024-10-01 00:00:00", "2024-10-31 23:59:59"),
        in_steps: int = 12,
        out_steps: int = 12,
        batch_size: int = 64,
        num_workers: int = 0,
        shuffle_training: bool = True,
        collate_fn: Callable = collate_fn,
        target_sensors: Optional[List[str]] = None,
        scale_method: Optional[Literal["normal", "strict", "none"]] = "normal",
        normalizer: Literal["minmax", "std"] = "std",
    ):
        # Call parent __init__ but we'll override steps_per_day since AGCRN doesn't use it
        super().__init__(
            dataset_path=dataset_path,
            training_period=training_period,
            validation_period=validation_period,
            test_period=test_period,
            in_steps=in_steps,
            out_steps=out_steps,
            steps_per_day=288,  # Not used by AGCRN
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle_training=shuffle_training,
            collate_fn=collate_fn,
            target_sensors=target_sensors,
            scale_method=scale_method,
        )
        
        self.normalizer = normalizer
        
        # Override the scaler attribute
        self._scaler: Optional[MinMaxScaler | StandardScaler] = None
        
        # Dataset placeholders (override parent's with correct type hints)
        self.train_dataset: Optional[AGCRNDataset] = None
        self.val_dataset: Optional[AGCRNDataset] = None
        self.test_dataset: Optional[AGCRNDataset] = None
    
    def _prepare_scaler(self, train_df: pd.DataFrame) -> None:
        """Prepare and fit the scaler on training data.
        
        Overrides parent method to support StandardScaler for AGCRN.
        
        Args:
            train_df: Training DataFrame
        """
        if self.scale_method is None or self.scale_method == "none":
            logger.info("Skipping scaler preparation (scale_method is None or 'none')")
            self._scaler = None
            return
        
        if self.scale_method == "strict":
            # Use only actual training samples for fitting
            temp_dataset = AGCRNDataset(
                train_df,
                in_steps=self.in_steps,
                out_steps=self.out_steps,
            )
            ref_data = self._get_strict_scaler_data(temp_dataset)
        else:
            # Use all training period data (excluding NaN)
            ref_data = train_df.values.reshape(-1, 1)
            ref_data = ref_data[~np.isnan(ref_data).any(axis=1)]
        
        # Choose scaler based on normalizer type
        if self.normalizer == "std":
            logger.info(f"Fitting StandardScaler with {len(ref_data)} data points using '{self.scale_method}' method")
            self._scaler = StandardScaler()
        else:
            logger.info(f"Fitting MinMaxScaler with {len(ref_data)} data points using '{self.scale_method}' method")
            self._scaler = MinMaxScaler(feature_range=(0, 1))
        
        self._scaler.fit(ref_data)
    
    def _get_strict_scaler_data(self, dataset: AGCRNDataset) -> np.ndarray:
        """Extract data for strict scaling method.
        
        Args:
            dataset: AGCRNDataset instance
            
        Returns:
            Array of shape (n_values, 1) for scaler fitting
        """
        data_list = []
        for i in tqdm(range(len(dataset)), desc="Extracting scaler reference data"):
            x, y = dataset[i]
            # Traffic values from x and y
            data_list.append(x[:, :, 0].numpy().flatten())
            data_list.append(y[:, :, 0].numpy().flatten())
        
        return np.concatenate(data_list).reshape(-1, 1)
    
    def setup(self, stage: Optional[Literal["fit", "validate", "test", "predict"]] = None) -> None:
        """Setup datasets for training/validation/testing.
        
        Overrides parent method to create AGCRNDataset instances.
        
        Args:
            stage: Current stage (fit, validate, test, predict)
        """
        logger.info(f"Setting up AGCRN data for stage: {stage}")
        
        # Load and split data (reuse parent method)
        train_df, val_df, test_df = self._load_and_split_data()
        
        # Store metadata
        self.num_nodes = train_df.shape[1]
        self.sensor_ids = list(train_df.columns)
        
        logger.info(f"Number of nodes: {self.num_nodes}")
        
        # Create AGCRN datasets (no temporal features)
        logger.info("Creating AGCRN datasets...")
        
        self.train_dataset = AGCRNDataset(
            train_df,
            in_steps=self.in_steps,
            out_steps=self.out_steps,
        )
        
        self.val_dataset = AGCRNDataset(
            val_df,
            in_steps=self.in_steps,
            out_steps=self.out_steps,
        )
        
        self.test_dataset = AGCRNDataset(
            test_df,
            in_steps=self.in_steps,
            out_steps=self.out_steps,
        )
        
        # Prepare and apply scaling
        self._prepare_scaler(train_df)
        self._apply_scaling(self.train_dataset, self.val_dataset, self.test_dataset)
        
        logger.info(f"Setup complete - Train: {len(self.train_dataset)}, "
                   f"Val: {len(self.val_dataset)}, Test: {len(self.test_dataset)}")
    
    def _apply_scaling(self, *datasets: AGCRNDataset) -> None:
        """Apply scaling to datasets.
        
        Args:
            datasets: AGCRNDataset instances to scale
        """
        if self.scale_method is None or self.scale_method == "none":
            logger.info("Skipping scaling (scale_method is None or 'none')")
            return
        
        if self._scaler is None:
            raise ValueError("Scaler is not fitted. Call _prepare_scaler first.")
        
        logger.info(f"Applying {self.normalizer} scaling to {len(datasets)} datasets")
        for dataset in datasets:
            dataset.apply_scaler(self._scaler)
