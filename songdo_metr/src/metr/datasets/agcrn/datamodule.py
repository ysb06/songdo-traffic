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

from .dataloader import collate_fn, collate_fn_with_missing
from .dataset import AGCRNDataset

logger = logging.getLogger(__name__)


class AGCRNDataModule(MLCAFormerDataModule):
    """PyTorch Lightning DataModule for AGCRN.

    This DataModule inherits from MLCAFormerDataModule but creates AGCRNDataset
    instances which output only traffic values without temporal features.

    Training 데이터와 Test 데이터를 별도 파일에서 로드하며,
    Training 데이터는 비율에 따라 train/validation으로 분할합니다.

    AGCRN learns spatial relationships through adaptive graph convolution
    using learnable node embeddings, so it doesn't require ToD/DoW features.

    Args:
        training_dataset_path: Path to the training HDF5 dataset file
        test_dataset_path: Path to the test HDF5 dataset file
        test_missing_path: Path to the test missing mask HDF5 file
        train_val_split: Train/Validation split ratio (default: 0.8 = 80% train, 20% val)
        in_steps: Number of input time steps (default: 12)
        out_steps: Number of output time steps (default: 12)
        batch_size: Batch size for DataLoader (default: 64)
        num_workers: Number of DataLoader workers (default: 0)
        shuffle_training: Whether to shuffle training data (default: True)
        collate_fn: Collate function for train/val DataLoader (default: collate_fn)
        target_sensors: List of sensor IDs to use (default: None = all sensors)
        scale_method: Scaling method - "normal", "strict", or "none" (default: "normal")
        normalizer: Normalizer type - "minmax" or "std" (default: "std")

    Data Shapes:
        Input (x): (batch_size, in_steps, n_vertex, 1)
        Target (y): (batch_size, out_steps, n_vertex, 1)
    """

    def __init__(
        self,
        training_dataset_path: str,
        test_dataset_path: str,
        test_missing_path: str,
        train_val_split: float = 0.8,
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
        # Call parent __init__
        super().__init__(
            training_dataset_path=training_dataset_path,
            test_dataset_path=test_dataset_path,
            test_missing_path=test_missing_path,
            train_val_split=train_val_split,
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

        # Dataset placeholders
        self.train_dataset: Optional[AGCRNDataset] = None
        self.val_dataset: Optional[AGCRNDataset] = None
        self.test_dataset: Optional[AGCRNDataset] = None

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

        logger.info(f"Applying scaling to {len(datasets)} datasets")
        for dataset in datasets:
            dataset.apply_scaler(self._scaler)

    def setup(
        self, stage: Optional[Literal["fit", "validate", "test", "predict"]] = None
    ) -> None:
        """Setup datasets for training/validation/testing.

        Overrides parent method to create AGCRNDataset instances.

        Args:
            stage: Current stage (fit, validate, test, predict)
        """
        logger.info(f"Setting up AGCRN data for stage: {stage}")

        # Load training data and split into train/val (reuse parent method)
        train_df, val_df = self._load_training_data()

        # Load test data and missing mask (reuse parent method)
        test_df, test_missing_mask = self._load_test_data()
        self.test_missing_mask = test_missing_mask

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
            missing_mask=test_missing_mask,
        )

        # Prepare and apply scaling
        self._prepare_scaler(train_df)
        self._apply_scaling(self.train_dataset, self.val_dataset, self.test_dataset)

        logger.info(
            f"Setup complete - Train: {len(self.train_dataset)}, "
            f"Val: {len(self.val_dataset)}, Test: {len(self.test_dataset)}"
        )
