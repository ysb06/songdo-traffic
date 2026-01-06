"""
Self-Sampling Index Generation for Phase 4.

This module generates self-sampling indices based on predicted labels.
For each sample, it finds similar past patterns (with the same predicted label)
to be used as reference samples in TASSGN training.
"""
import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)


def generate_self_sampling_index(
    pred_label: np.ndarray,
    num_samples: int = 7,
    history_len: int = 12,
    future_len: int = 12,
) -> np.ndarray:
    """Generate self-sampling indices based on predicted labels.
    
    For each sample and each node, find the most recent past samples
    that share the same predicted label. This allows the model to
    reference similar historical patterns during prediction.
    
    IMPORTANT: Only samples that appear BEFORE the current sample are
    selected to avoid information leakage.
    
    Args:
        pred_label: Predicted labels, shape (total_samples, num_nodes, 1)
        num_samples: Number of similar samples to select per node (default: 7)
        history_len: Length of history series (default: 12)
        future_len: Length of future series (default: 12)
        
    Returns:
        sample_index: Self-sampling indices, shape (total_samples', num_nodes, num_samples, 1)
                     Note: total_samples' may be smaller than total_samples if 
                     history_len > future_len (early samples can't be generated)
                     
    Example:
        ```python
        pred_labels = np.load("predicted_label.npy")  # (17833, 170, 1)
        sample_index = generate_self_sampling_index(
            pred_labels, 
            num_samples=7,
            history_len=12,
            future_len=12
        )
        # sample_index shape: (17833, 170, 7, 1)
        ```
    """
    total_size, num_nodes, _ = pred_label.shape
    pred_label = pred_label.astype(np.int32)
    
    logger.info(f"Generating self-sampling index...")
    logger.info(f"Predicted label shape: {pred_label.shape}")
    logger.info(f"num_samples: {num_samples}, history_len: {history_len}, future_len: {future_len}")
    
    # Maximum label value (for cluster initialization)
    max_label = np.max(pred_label) + 1
    logger.info(f"Number of unique labels: {max_label}")
    
    # Initialize cluster storage for each node and label
    # clusters[node_id][label] = list of sample indices with that label
    clusters = [[[] for _ in range(max_label)] for _ in range(num_nodes)]
    
    sample_index_list = []
    
    # Process each time step
    for t in tqdm(range(total_size), desc="Generating sample indices"):
        # Skip early samples where we can't generate valid history
        # (when history series would extend before the start of data)
        if t + history_len - future_len < 0:
            continue
        
        node_sample_index = []
        
        for node_id in range(num_nodes):
            label = pred_label[t, node_id, 0]
            
            # Index explanation:
            # The window_data is reshaped as (total_time_steps * num_nodes, future_len, features)
            # So 2D index (time_t, node_id) maps to 1D index (time_t * num_nodes + node_id)
            
            # Find past samples with the same label
            # Only use samples that appear BEFORE current sample to avoid information leakage
            max_valid_time = (t + history_len - future_len) * num_nodes + node_id
            
            similar_samples = [
                idx for idx in clusters[node_id][label] 
                if idx <= max_valid_time
            ]
            
            # If no similar samples found, use the most recent history
            if len(similar_samples) == 0:
                similar_samples.append(max_valid_time)
            
            # Expand list if too short by repeating
            while len(similar_samples) < num_samples:
                similar_samples = similar_samples + similar_samples
            
            # Take the most recent num_samples
            selected_samples = similar_samples[-num_samples:]
            node_sample_index.append(selected_samples)
            
            # Add current sample to cluster for future reference
            current_idx = (t + history_len) * num_nodes + node_id
            clusters[node_id][label].append(current_idx)
        
        sample_index_list.append(node_sample_index)
    
    # Convert to numpy array
    sample_index = np.array(sample_index_list, dtype=np.int64)
    # Shape: (total_samples', num_nodes, num_samples)
    
    # Add trailing dimension
    sample_index = sample_index.reshape(-1, num_nodes, num_samples, 1)
    
    logger.info(f"Self-sampling index shape: {sample_index.shape}")
    
    return sample_index


def generate_train_val_test_index(
    sample_index: np.ndarray,
    total_size: int,
    train_ratio: float = 0.6,
    validation_ratio: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split self-sampling index into train/val/test sets.
    
    Args:
        sample_index: Self-sampling indices, shape (total_samples', num_nodes, num_samples, 1)
        total_size: Original total number of samples (before any truncation)
        train_ratio: Training set ratio (default: 0.6)
        validation_ratio: Validation set ratio (default: 0.2)
        
    Returns:
        Tuple of (train_index, val_index, test_index)
        
    Example:
        ```python
        train_idx, val_idx, test_idx = generate_train_val_test_index(
            sample_index,
            total_size=17833,
            train_ratio=0.6,
            validation_ratio=0.2
        )
        ```
    """
    test_ratio = 1 - train_ratio - validation_ratio
    
    assert 0 < train_ratio < 1, f"Invalid train_ratio: {train_ratio}"
    assert 0 < validation_ratio < 1, f"Invalid validation_ratio: {validation_ratio}"
    assert 0 < test_ratio < 1, f"Invalid test_ratio: {test_ratio}"
    
    # Calculate sizes based on total_size
    val_size = int(total_size * validation_ratio)
    test_size = int(total_size * test_ratio)
    
    # Adjust train_size based on actual sample_index length
    # (may be smaller if history_len > future_len)
    train_size = sample_index.shape[0] - val_size - test_size
    
    if train_size <= 0:
        raise ValueError(
            f"Invalid split: sample_index has {sample_index.shape[0]} samples, "
            f"but val_size={val_size}, test_size={test_size}"
        )
    
    # Split
    train_index = sample_index[:train_size]
    val_index = sample_index[train_size:train_size + val_size]
    test_index = sample_index[train_size + val_size:]
    
    logger.info(f"Split sizes - Train: {len(train_index)}, Val: {len(val_index)}, Test: {len(test_index)}")
    
    return train_index, val_index, test_index


def generate_window_data(
    data: np.ndarray,
    tod: np.ndarray,
    dow: np.ndarray,
    future_len: int = 12,
) -> np.ndarray:
    """Generate window data for self-sampling.
    
    Creates all possible future windows from the traffic data.
    
    Args:
        data: Traffic data, shape (time_steps, num_nodes)
        tod: Time-of-Day array, shape (time_steps,)
        dow: Day-of-Week array, shape (time_steps,)
        future_len: Length of future series (default: 12)
        
    Returns:
        window_data: All future windows, shape (num_windows, future_len, num_nodes, 3)
                    Features: [traffic, tod, dow]
                    
    Example:
        ```python
        window_data = generate_window_data(
            data=traffic_values,  # (17856, 170)
            tod=time_of_day,      # (17856,)
            dow=day_of_week,      # (17856,)
            future_len=12
        )
        # window_data shape: (17845, 12, 170, 3)
        ```
    """
    time_steps, num_nodes = data.shape
    num_windows = time_steps - future_len + 1
    
    if num_windows <= 0:
        raise ValueError(f"Data length ({time_steps}) is shorter than future_len ({future_len})")
    
    logger.info(f"Generating {num_windows} windows...")
    
    window_data = np.zeros((num_windows, future_len, num_nodes, 3), dtype=np.float32)
    
    for t in range(num_windows):
        # Traffic values
        window_data[t, :, :, 0] = data[t:t + future_len, :]
        # Time-of-Day (broadcast to all nodes)
        window_data[t, :, :, 1] = tod[t:t + future_len, np.newaxis]
        # Day-of-Week (broadcast to all nodes)
        window_data[t, :, :, 2] = dow[t:t + future_len, np.newaxis]
    
    logger.info(f"Window data shape: {window_data.shape}")
    
    return window_data


def generate_sampling_artifacts(
    predicted_label_path: str,
    output_dir: str,
    num_samples: int = 7,
    history_len: int = 12,
    future_len: int = 12,
    train_ratio: float = 0.6,
    validation_ratio: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convenience function to generate all Phase 4 artifacts.
    
    Loads predicted labels, generates self-sampling indices,
    and splits into train/val/test sets.
    
    Args:
        predicted_label_path: Path to predicted_label.npy from Phase 3
        output_dir: Directory to save output files
        num_samples: Number of samples per node (default: 7)
        history_len: Length of history series (default: 12)
        future_len: Length of future series (default: 12)
        train_ratio: Training set ratio (default: 0.6)
        validation_ratio: Validation set ratio (default: 0.2)
        
    Returns:
        Tuple of (train_sample_index, val_sample_index, test_sample_index)
        
    Example:
        ```python
        generate_sampling_artifacts(
            predicted_label_path="./data/predicted_label.npy",
            output_dir="./data",
            num_samples=7,
        )
        # Creates: train_sample_index.npy, val_sample_index.npy, test_sample_index.npy
        ```
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load predicted labels
    logger.info(f"Loading predicted labels from {predicted_label_path}")
    pred_labels = np.load(predicted_label_path)
    logger.info(f"Predicted labels shape: {pred_labels.shape}")
    
    total_size = pred_labels.shape[0]
    
    # Generate self-sampling index
    sample_index = generate_self_sampling_index(
        pred_labels,
        num_samples=num_samples,
        history_len=history_len,
        future_len=future_len,
    )
    
    # Split into train/val/test
    train_index, val_index, test_index = generate_train_val_test_index(
        sample_index,
        total_size=total_size,
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
    )
    
    # Save
    train_path = output_path / "train_sample_index.npy"
    val_path = output_path / "val_sample_index.npy"
    test_path = output_path / "test_sample_index.npy"
    
    np.save(train_path, train_index)
    np.save(val_path, val_index)
    np.save(test_path, test_index)
    
    logger.info(f"Saved train_sample_index.npy: {train_index.shape}")
    logger.info(f"Saved val_sample_index.npy: {val_index.shape}")
    logger.info(f"Saved test_sample_index.npy: {test_index.shape}")
    
    return train_index, val_index, test_index


def generate_window_data_from_datamodule(
    dataset_path: str,
    output_dir: str,
    future_len: int = 12,
    steps_per_day: int = 288,
    target_sensors: Optional[list] = None,
    scaler=None,
) -> np.ndarray:
    """Generate and save window data from HDF5 dataset.
    
    This function loads the full dataset and generates window data
    for self-sampling in Phase 5.
    
    Args:
        dataset_path: Path to HDF5 dataset file
        output_dir: Directory to save window_data.npy
        future_len: Length of future series (default: 12)
        steps_per_day: Number of steps per day (default: 288)
        target_sensors: List of sensor IDs to use (default: None = all)
        scaler: Optional fitted scaler for normalization
        
    Returns:
        window_data: All future windows, shape (num_windows, future_len, num_nodes, 3)
        
    Example:
        ```python
        window_data = generate_window_data_from_datamodule(
            dataset_path="metr-imc.h5",
            output_dir="./data",
            future_len=12,
        )
        ```
    """
    from metr.components.metr_imc.traffic_data import get_raw
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load data
    logger.info(f"Loading data from {dataset_path}")
    raw = get_raw(dataset_path)
    raw_df = raw.data
    
    # Filter sensors if specified
    if target_sensors is not None:
        raw_df = raw_df.loc[:, target_sensors]
    
    # Extract arrays
    data = raw_df.values.astype(np.float32)
    
    # Apply scaler if provided
    if scaler is not None:
        data_flat = data.reshape(-1, 1)
        data_scaled = scaler.transform(data_flat)
        data = data_scaled.reshape(data.shape)
    
    # Extract temporal features (TASSGN format: integers)
    index = pd.DatetimeIndex(raw_df.index)
    minutes_per_step = 24 * 60 / steps_per_day
    minutes = index.hour * 60 + index.minute
    tod = (minutes / minutes_per_step).astype(np.int32).values
    dow = index.dayofweek.values.astype(np.int32)
    
    # Generate window data
    window_data = generate_window_data(data, tod, dow, future_len)
    
    # Save
    output_file = output_path / "window_data.npy"
    np.save(output_file, window_data)
    
    logger.info(f"Saved window_data.npy: {window_data.shape}")
    
    return window_data
