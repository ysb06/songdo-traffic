"""
TASSGN DataModule classes for PyTorch Lightning.

This module provides DataModule classes for each phase of TASSGN training:
- EncoderDataModule: Phase 1 - STIDEncoder pre-training
- PredictorDataModule: Phase 3 - Predictor training  
- TASSGNDataModule: Phase 5 - Final TASSGN model training
"""
import logging
from pathlib import Path
from typing import Callable, List, Literal, Optional, Tuple, Union

import lightning as L
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from metr.components.metr_imc.traffic_data import get_raw

from .dataloader import encoder_collate_fn, predictor_collate_fn, tassgn_collate_fn
from .dataset import (
    EncoderDataset,
    PredictorDataset,
    TASSGNDataset,
    TASSGNBaseDataset,
    WindowDataset,
)

logger = logging.getLogger(__name__)


class TASSGNBaseDataModule(L.LightningDataModule):
    """Base DataModule with common functionality for TASSGN phases.
    
    Provides:
    - Data loading from HDF5
    - Period-based train/val/test splitting
    - Standard scaling (TASSGN uses StandardScaler, not MinMax)
    - Metadata storage (num_nodes, sensor_ids)
    
    Args:
        dataset_path: Path to the HDF5 dataset file
        training_period: Tuple of (start_datetime, end_datetime) for training
        validation_period: Tuple of (start_datetime, end_datetime) for validation
        test_period: Tuple of (start_datetime, end_datetime) for testing
        in_steps: Number of input time steps (default: 12)
        out_steps: Number of output time steps (default: 12)
        steps_per_day: Number of time steps per day (default: 288)
        batch_size: Batch size for DataLoader (default: 32)
        num_workers: Number of DataLoader workers (default: 0)
        target_sensors: List of sensor IDs to use (default: None = all sensors)
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
        batch_size: int = 32,
        num_workers: int = 0,
        target_sensors: Optional[List[str]] = None,
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
        self.target_sensors = target_sensors
        
        self._scaler: Optional[StandardScaler] = None
        
        # Metadata
        self.num_nodes: Optional[int] = None
        self.sensor_ids: Optional[List[str]] = None
    
    @property
    def scaler(self) -> Optional[StandardScaler]:
        """Get the fitted scaler."""
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
        """Prepare and fit StandardScaler on training data.
        
        TASSGN uses StandardScaler (z-score normalization) instead of MinMaxScaler.
        
        Args:
            train_df: Training DataFrame
        """
        # Flatten and remove NaN for scaler fitting
        ref_data = train_df.values.reshape(-1, 1)
        ref_data = ref_data[~np.isnan(ref_data).any(axis=1)]
        
        logger.info(f"Fitting StandardScaler with {len(ref_data)} data points")
        self._scaler = StandardScaler()
        self._scaler.fit(ref_data)
        
        assert self._scaler.mean_ is not None and self._scaler.scale_ is not None
        logger.info(f"Scaler stats - mean: {self._scaler.mean_[0]:.4f}, std: {self._scaler.scale_[0]:.4f}")
    
    def _apply_scaling(self, *datasets: TASSGNBaseDataset) -> None:
        """Apply scaling to datasets.
        
        Args:
            datasets: Dataset instances to scale
        """
        if self._scaler is None:
            raise ValueError("Scaler is not fitted. Call _prepare_scaler first.")
        
        logger.info(f"Applying StandardScaler to {len(datasets)} datasets")
        for dataset in datasets:
            dataset.apply_scaler(self._scaler)
    
    def _create_dataloader(
        self, 
        dataset, 
        shuffle: bool, 
        collate_fn: Callable
    ) -> DataLoader:
        """Create a DataLoader with common settings.
        
        Args:
            dataset: Dataset instance
            shuffle: Whether to shuffle data
            collate_fn: Collate function
            
        Returns:
            DataLoader instance
        """
        kwargs = {
            "batch_size": self.batch_size,
            "shuffle": shuffle,
            "collate_fn": collate_fn,
        }
        
        if self.num_workers > 0:
            kwargs["num_workers"] = self.num_workers
            kwargs["persistent_workers"] = True
        
        return DataLoader(dataset, **kwargs)


class EncoderDataModule(TASSGNBaseDataModule):
    """DataModule for Phase 1: STIDEncoder pre-training.
    
    STIDEncoder learns to reconstruct masked future time series.
    This DataModule provides future series data (y_data) only.
    
    After training, use `save_representations()` to generate and save
    the encoded representations for Phase 2 (Labeler).
    
    Args:
        dataset_path: Path to the HDF5 dataset file
        training_period: Tuple of (start_datetime, end_datetime) for training
        validation_period: Tuple of (start_datetime, end_datetime) for validation
        test_period: Optional tuple for testing (usually not needed for encoder)
        out_steps: Length of future series (default: 12)
        steps_per_day: Number of time steps per day (default: 288)
        batch_size: Batch size for DataLoader (default: 32)
        num_workers: Number of DataLoader workers (default: 0)
        target_sensors: List of sensor IDs to use (default: None = all sensors)
        
    Example:
        ```python
        dm = EncoderDataModule(
            dataset_path="metr-imc.h5",
            training_period=("2023-01-01", "2023-12-31"),
            validation_period=("2024-01-01", "2024-03-31"),
        )
        dm.setup()
        
        # Train encoder...
        trainer.fit(encoder_model, dm)
        
        # Save representations for Phase 2
        dm.save_representations(encoder_model, output_dir="./data")
        ```
    """
    
    def __init__(
        self,
        dataset_path: str,
        training_period: Tuple[str, str] = ("2022-11-01 00:00:00", "2024-07-31 23:59:59"),
        validation_period: Tuple[str, str] = ("2024-08-01 00:00:00", "2024-09-30 23:59:59"),
        test_period: Optional[Tuple[str, str]] = None,
        out_steps: int = 12,
        steps_per_day: int = 288,
        batch_size: int = 32,
        num_workers: int = 0,
        target_sensors: Optional[List[str]] = None,
        shuffle_training: bool = True,
    ):
        # For encoder, in_steps is same as out_steps (we use y_data)
        super().__init__(
            dataset_path=dataset_path,
            training_period=training_period,
            validation_period=validation_period,
            test_period=test_period or validation_period,  # Fallback
            in_steps=out_steps,  # Same as out_steps for encoder
            out_steps=out_steps,
            steps_per_day=steps_per_day,
            batch_size=batch_size,
            num_workers=num_workers,
            target_sensors=target_sensors,
        )
        self.shuffle_training = shuffle_training
        
        # Dataset placeholders
        self.train_dataset: Optional[EncoderDataset] = None
        self.val_dataset: Optional[EncoderDataset] = None
    
    def setup(self, stage: Optional[Literal["fit", "validate", "test", "predict"]] = None) -> None:
        """Setup datasets for training/validation.
        
        Args:
            stage: Current stage
        """
        logger.info(f"Setting up EncoderDataModule for stage: {stage}")
        
        # Load and split data
        train_df, val_df, _ = self._load_and_split_data()
        
        # Store metadata
        self.num_nodes = train_df.shape[1]
        self.sensor_ids = list(train_df.columns)
        
        logger.info(f"Number of nodes: {self.num_nodes}")
        
        # Create datasets
        logger.info("Creating EncoderDatasets...")
        
        self.train_dataset = EncoderDataset(
            train_df,
            in_steps=self.in_steps,
            out_steps=self.out_steps,
            steps_per_day=self.steps_per_day,
        )
        
        self.val_dataset = EncoderDataset(
            val_df,
            in_steps=self.in_steps,
            out_steps=self.out_steps,
            steps_per_day=self.steps_per_day,
        )
        
        # Prepare and apply scaling
        self._prepare_scaler(train_df)
        self._apply_scaling(self.train_dataset, self.val_dataset)
        
        logger.info(f"Setup complete - Train: {len(self.train_dataset)}, Val: {len(self.val_dataset)}")
    
    def train_dataloader(self) -> DataLoader:
        """Create training DataLoader."""
        if self.train_dataset is None:
            raise RuntimeError("train_dataset is None. Call setup() first.")
        
        return self._create_dataloader(
            self.train_dataset, 
            shuffle=self.shuffle_training,
            collate_fn=encoder_collate_fn
        )
    
    def val_dataloader(self) -> DataLoader:
        """Create validation DataLoader."""
        if self.val_dataset is None:
            raise RuntimeError("val_dataset is None. Call setup() first.")
        
        return self._create_dataloader(
            self.val_dataset,
            shuffle=False,
            collate_fn=encoder_collate_fn
        )
    
    def save_representations(
        self, 
        encoder: torch.nn.Module, 
        output_dir: str,
        device: str = "cpu"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate and save encoded representations for Phase 2.
        
        Args:
            encoder: Trained STIDEncoder model
            output_dir: Directory to save representation files
            device: Device to use for inference
            
        Returns:
            Tuple of (train_representation, val_representation) arrays
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        encoder.eval()
        encoder.to(device)
        
        if self.train_dataset is None or self.val_dataset is None:
            raise RuntimeError("Datasets not initialized. Call setup() first.")
        
        def encode_dataset(dataset: EncoderDataset) -> np.ndarray:
            """Encode all samples in a dataset."""
            representations = []
            
            loader = DataLoader(
                dataset, 
                batch_size=self.batch_size, 
                shuffle=False,
                collate_fn=encoder_collate_fn
            )
            
            with torch.no_grad():
                for y_batch, _ in tqdm(loader, desc="Encoding"):
                    y_batch = y_batch.to(device)
                    # STIDEncoder.encode() returns (B, hidden_dim*4, N, 1)
                    repr_batch = encoder.encode(y_batch, mask=False)  # type: ignore
                    representations.append(repr_batch.cpu().numpy())
            
            return np.concatenate(representations, axis=0)
        
        logger.info("Generating train representations...")
        train_repr = encode_dataset(self.train_dataset)
        
        logger.info("Generating val representations...")
        val_repr = encode_dataset(self.val_dataset)
        
        # Save
        train_path = output_path / "train_representation.npy"
        val_path = output_path / "val_representation.npy"
        
        np.save(train_path, train_repr)
        np.save(val_path, val_repr)
        
        logger.info(f"Saved train_representation.npy: {train_repr.shape}")
        logger.info(f"Saved val_representation.npy: {val_repr.shape}")
        
        return train_repr, val_repr


class PredictorDataModule(TASSGNBaseDataModule):
    """DataModule for Phase 3: Predictor training.
    
    Predictor learns to predict future pattern labels from history series.
    Requires cluster labels generated in Phase 2.
    
    Args:
        dataset_path: Path to the HDF5 dataset file
        train_cluster_label_path: Path to train_cluster_label.npy from Phase 2
        val_cluster_label_path: Path to val_cluster_label.npy from Phase 2
        training_period: Tuple of (start_datetime, end_datetime) for training
        validation_period: Tuple of (start_datetime, end_datetime) for validation
        in_steps: Number of input time steps (default: 12)
        out_steps: Number of output time steps (default: 12)
        steps_per_day: Number of time steps per day (default: 288)
        batch_size: Batch size for DataLoader (default: 32)
        num_workers: Number of DataLoader workers (default: 0)
        target_sensors: List of sensor IDs to use (default: None = all sensors)
        
    Example:
        ```python
        dm = PredictorDataModule(
            dataset_path="metr-imc.h5",
            train_cluster_label_path="./data/train_cluster_label.npy",
            val_cluster_label_path="./data/val_cluster_label.npy",
        )
        dm.setup()
        
        # Train predictor...
        trainer.fit(predictor_model, dm)
        
        # Save predicted labels for Phase 4
        dm.save_predicted_labels(predictor_model, output_dir="./data")
        ```
    """
    
    def __init__(
        self,
        dataset_path: str,
        train_cluster_label_path: str,
        val_cluster_label_path: str,
        training_period: Tuple[str, str] = ("2022-11-01 00:00:00", "2024-07-31 23:59:59"),
        validation_period: Tuple[str, str] = ("2024-08-01 00:00:00", "2024-09-30 23:59:59"),
        test_period: Optional[Tuple[str, str]] = ("2024-10-01 00:00:00", "2024-10-31 23:59:59"),
        in_steps: int = 12,
        out_steps: int = 12,
        steps_per_day: int = 288,
        batch_size: int = 32,
        num_workers: int = 0,
        target_sensors: Optional[List[str]] = None,
        shuffle_training: bool = True,
    ):
        super().__init__(
            dataset_path=dataset_path,
            training_period=training_period,
            validation_period=validation_period,
            test_period=test_period or validation_period,
            in_steps=in_steps,
            out_steps=out_steps,
            steps_per_day=steps_per_day,
            batch_size=batch_size,
            num_workers=num_workers,
            target_sensors=target_sensors,
        )
        
        self.train_cluster_label_path = train_cluster_label_path
        self.val_cluster_label_path = val_cluster_label_path
        self.shuffle_training = shuffle_training
        
        # Dataset placeholders
        self.train_dataset: Optional[PredictorDataset] = None
        self.val_dataset: Optional[PredictorDataset] = None
        
        # Number of clusters (for model output dimension)
        self.num_clusters: Optional[int] = None
    
    def setup(self, stage: Optional[Literal["fit", "validate", "test", "predict"]] = None) -> None:
        """Setup datasets for training/validation.
        
        Args:
            stage: Current stage
        """
        logger.info(f"Setting up PredictorDataModule for stage: {stage}")
        
        # Load cluster labels
        logger.info(f"Loading cluster labels...")
        train_labels = np.load(self.train_cluster_label_path)
        val_labels = np.load(self.val_cluster_label_path)
        
        self.num_clusters = int(np.max(train_labels)) + 1
        logger.info(f"Number of clusters: {self.num_clusters}")
        
        # Load and split data
        train_df, val_df, _ = self._load_and_split_data()
        
        # Store metadata
        self.num_nodes = train_df.shape[1]
        self.sensor_ids = list(train_df.columns)
        
        logger.info(f"Number of nodes: {self.num_nodes}")
        
        # Create datasets
        logger.info("Creating PredictorDatasets...")
        
        self.train_dataset = PredictorDataset(
            train_df,
            cluster_labels=train_labels,
            in_steps=self.in_steps,
            out_steps=self.out_steps,
            steps_per_day=self.steps_per_day,
        )
        
        self.val_dataset = PredictorDataset(
            val_df,
            cluster_labels=val_labels,
            in_steps=self.in_steps,
            out_steps=self.out_steps,
            steps_per_day=self.steps_per_day,
        )
        
        # Prepare and apply scaling
        self._prepare_scaler(train_df)
        self._apply_scaling(self.train_dataset, self.val_dataset)
        
        logger.info(f"Setup complete - Train: {len(self.train_dataset)}, Val: {len(self.val_dataset)}")
    
    def train_dataloader(self) -> DataLoader:
        """Create training DataLoader."""
        if self.train_dataset is None:
            raise RuntimeError("train_dataset is None. Call setup() first.")
        
        return self._create_dataloader(
            self.train_dataset,
            shuffle=self.shuffle_training,
            collate_fn=predictor_collate_fn
        )
    
    def val_dataloader(self) -> DataLoader:
        """Create validation DataLoader."""
        if self.val_dataset is None:
            raise RuntimeError("val_dataset is None. Call setup() first.")
        
        return self._create_dataloader(
            self.val_dataset,
            shuffle=False,
            collate_fn=predictor_collate_fn
        )
    
    def save_predicted_labels(
        self, 
        predictor: torch.nn.Module, 
        output_dir: str,
        device: str = "cpu"
    ) -> np.ndarray:
        """Generate and save predicted labels for all data (train+val+test).
        
        Used in Phase 4 for self-sampling index generation.
        
        Args:
            predictor: Trained Predictor model
            output_dir: Directory to save predicted labels
            device: Device to use for inference
            
        Returns:
            Predicted labels array of shape (total_samples, n_vertex, 1)
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        predictor.eval()
        predictor.to(device)
        
        # Load full data (train + val + test)
        train_df, val_df, test_df = self._load_and_split_data()
        full_df = pd.concat([train_df, val_df, test_df])
        
        # Create temporary dataset for full data
        from .dataset import TASSGNBaseDataset
        
        class TempDataset(TASSGNBaseDataset):
            def __getitem__(self, idx):
                t = self.valid_indices[idx]
                x = self._get_x_features(t)
                return torch.from_numpy(x)
        
        temp_dataset = TempDataset(
            full_df,
            in_steps=self.in_steps,
            out_steps=self.out_steps,
            steps_per_day=self.steps_per_day,
        )
        if self._scaler is not None:
            temp_dataset.apply_scaler(self._scaler)
        
        # Simple collate function
        def simple_collate(batch):
            return torch.stack(batch, dim=0)
        
        loader = DataLoader(
            temp_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=simple_collate
        )
        
        # Predict labels
        pred_labels = []
        
        with torch.no_grad():
            for x_batch in tqdm(loader, desc="Predicting labels"):
                x_batch = x_batch.to(device)
                logits = predictor(x_batch)  # (B, N, num_clusters)
                labels = torch.argmax(
                    torch.softmax(logits, dim=-1), 
                    dim=-1
                ).unsqueeze(-1)  # (B, N, 1)
                pred_labels.append(labels.cpu().numpy())
        
        pred_labels = np.concatenate(pred_labels, axis=0)
        
        # Save
        output_file = output_path / "predicted_label.npy"
        np.save(output_file, pred_labels)
        
        logger.info(f"Saved predicted_label.npy: {pred_labels.shape}")
        
        return pred_labels


class TASSGNDataModule(TASSGNBaseDataModule):
    """DataModule for Phase 5: Final TASSGN model training.
    
    TASSGN model takes history series, self-sampled similar patterns,
    and predicts future traffic values.
    
    Requires:
    - sample_index files from Phase 4
    - window_data for self-sampling
    
    Args:
        dataset_path: Path to the HDF5 dataset file
        train_sample_index_path: Path to train_sample_index.npy from Phase 4
        val_sample_index_path: Path to val_sample_index.npy from Phase 4
        test_sample_index_path: Path to test_sample_index.npy from Phase 4
        window_data_path: Path to window_data.npy
        training_period: Tuple of (start_datetime, end_datetime) for training
        validation_period: Tuple of (start_datetime, end_datetime) for validation
        test_period: Tuple of (start_datetime, end_datetime) for testing
        in_steps: Number of input time steps (default: 12)
        out_steps: Number of output time steps (default: 12)
        steps_per_day: Number of time steps per day (default: 288)
        num_samples: Number of samples per node for self-sampling (default: 7)
        batch_size: Batch size for DataLoader (default: 32)
        num_workers: Number of DataLoader workers (default: 0)
        target_sensors: List of sensor IDs to use (default: None = all sensors)
        
    Example:
        ```python
        dm = TASSGNDataModule(
            dataset_path="metr-imc.h5",
            train_sample_index_path="./data/train_sample_index.npy",
            val_sample_index_path="./data/val_sample_index.npy",
            test_sample_index_path="./data/test_sample_index.npy",
            window_data_path="./data/window_data.npy",
        )
        dm.setup()
        
        # Train TASSGN
        trainer.fit(tassgn_model, dm)
        trainer.test(tassgn_model, dm)
        ```
    """
    
    def __init__(
        self,
        dataset_path: str,
        train_sample_index_path: str,
        val_sample_index_path: str,
        test_sample_index_path: str,
        window_data_path: str,
        training_period: Tuple[str, str] = ("2022-11-01 00:00:00", "2024-07-31 23:59:59"),
        validation_period: Tuple[str, str] = ("2024-08-01 00:00:00", "2024-09-30 23:59:59"),
        test_period: Tuple[str, str] = ("2024-10-01 00:00:00", "2024-10-31 23:59:59"),
        in_steps: int = 12,
        out_steps: int = 12,
        steps_per_day: int = 288,
        num_samples: int = 7,
        batch_size: int = 32,
        num_workers: int = 0,
        target_sensors: Optional[List[str]] = None,
        shuffle_training: bool = True,
    ):
        super().__init__(
            dataset_path=dataset_path,
            training_period=training_period,
            validation_period=validation_period,
            test_period=test_period,
            in_steps=in_steps,
            out_steps=out_steps,
            steps_per_day=steps_per_day,
            batch_size=batch_size,
            num_workers=num_workers,
            target_sensors=target_sensors,
        )
        
        self.train_sample_index_path = train_sample_index_path
        self.val_sample_index_path = val_sample_index_path
        self.test_sample_index_path = test_sample_index_path
        self.window_data_path = window_data_path
        self.num_samples = num_samples
        self.shuffle_training = shuffle_training
        
        # Dataset placeholders
        self.train_dataset: Optional[TASSGNDataset] = None
        self.val_dataset: Optional[TASSGNDataset] = None
        self.test_dataset: Optional[TASSGNDataset] = None
        
        # Window data (shared across datasets)
        self._window_data: Optional[np.ndarray] = None
    
    def setup(self, stage: Optional[Literal["fit", "validate", "test", "predict"]] = None) -> None:
        """Setup datasets for training/validation/testing.
        
        Args:
            stage: Current stage
        """
        logger.info(f"Setting up TASSGNDataModule for stage: {stage}")
        
        # Load sample indices
        logger.info("Loading sample indices...")
        train_sample_index = np.load(self.train_sample_index_path)
        val_sample_index = np.load(self.val_sample_index_path)
        test_sample_index = np.load(self.test_sample_index_path)
        
        logger.info(f"Sample index shapes - Train: {train_sample_index.shape}, "
                   f"Val: {val_sample_index.shape}, Test: {test_sample_index.shape}")
        
        # Load window data
        logger.info(f"Loading window data from {self.window_data_path}...")
        self._window_data = np.load(self.window_data_path)
        assert self._window_data is not None
        logger.info(f"Window data shape: {self._window_data.shape}")
        
        # Load and split data
        train_df, val_df, test_df = self._load_and_split_data()
        
        # Store metadata
        self.num_nodes = train_df.shape[1]
        self.sensor_ids = list(train_df.columns)
        
        logger.info(f"Number of nodes: {self.num_nodes}")
        
        # Prepare scaler first (need for window_data too)
        self._prepare_scaler(train_df)
        assert self._scaler is not None
        assert self._window_data is not None
        
        # Scale window data
        window_traffic = self._window_data[:, :, :, 0:1]
        window_traffic_flat = window_traffic.reshape(-1, 1)
        window_traffic_scaled = self._scaler.transform(window_traffic_flat)
        self._window_data[:, :, :, 0:1] = window_traffic_scaled.reshape(window_traffic.shape)
        
        # Create datasets
        logger.info("Creating TASSGNDatasets...")
        
        self.train_dataset = TASSGNDataset(
            train_df,
            sample_index=train_sample_index,
            window_data=self._window_data,
            in_steps=self.in_steps,
            out_steps=self.out_steps,
            steps_per_day=self.steps_per_day,
            num_samples=self.num_samples,
        )
        
        self.val_dataset = TASSGNDataset(
            val_df,
            sample_index=val_sample_index,
            window_data=self._window_data,
            in_steps=self.in_steps,
            out_steps=self.out_steps,
            steps_per_day=self.steps_per_day,
            num_samples=self.num_samples,
        )
        
        self.test_dataset = TASSGNDataset(
            test_df,
            sample_index=test_sample_index,
            window_data=self._window_data,
            in_steps=self.in_steps,
            out_steps=self.out_steps,
            steps_per_day=self.steps_per_day,
            num_samples=self.num_samples,
        )
        
        # Apply scaling to datasets
        self._apply_scaling(self.train_dataset, self.val_dataset, self.test_dataset)
        
        logger.info(f"Setup complete - Train: {len(self.train_dataset)}, "
                   f"Val: {len(self.val_dataset)}, Test: {len(self.test_dataset)}")
    
    def train_dataloader(self) -> DataLoader:
        """Create training DataLoader."""
        if self.train_dataset is None:
            raise RuntimeError("train_dataset is None. Call setup() first.")
        
        return self._create_dataloader(
            self.train_dataset,
            shuffle=self.shuffle_training,
            collate_fn=tassgn_collate_fn
        )
    
    def val_dataloader(self) -> DataLoader:
        """Create validation DataLoader."""
        if self.val_dataset is None:
            raise RuntimeError("val_dataset is None. Call setup() first.")
        
        return self._create_dataloader(
            self.val_dataset,
            shuffle=False,
            collate_fn=tassgn_collate_fn
        )
    
    def test_dataloader(self) -> DataLoader:
        """Create test DataLoader."""
        if self.test_dataset is None:
            raise RuntimeError("test_dataset is None. Call setup() first.")
        
        return self._create_dataloader(
            self.test_dataset,
            shuffle=False,
            collate_fn=tassgn_collate_fn
        )
