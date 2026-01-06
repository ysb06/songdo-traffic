"""
TASSGN dataset module for traffic prediction.

This module provides Dataset, DataModule, and utility classes for the TASSGN model,
which uses self-sampling and transition-aware graph learning for traffic forecasting.

The TASSGN training pipeline consists of 5 phases:
1. Phase 1 (EncoderDataModule): STIDEncoder pre-training for future pattern encoding
2. Phase 2 (Labeler): Binary clustering to generate pattern labels
3. Phase 3 (PredictorDataModule): Predictor training for label prediction
4. Phase 4 (sampling): Self-sampling index generation
5. Phase 5 (TASSGNDataModule): Final TASSGN model training

Example usage:
    ```python
    from metr.datasets.tassgn import (
        EncoderDataModule,
        PredictorDataModule,
        TASSGNDataModule,
        generate_cluster_labels,
        generate_sampling_artifacts,
    )
    
    # Phase 1: Encoder pre-training
    encoder_dm = EncoderDataModule(dataset_path="metr-imc.h5")
    encoder_dm.setup()
    # ... train encoder ...
    encoder_dm.save_representations(encoder, output_dir="./data")
    
    # Phase 2: Generate cluster labels
    generate_cluster_labels(
        train_representation_path="./data/train_representation.npy",
        val_representation_path="./data/val_representation.npy",
        output_dir="./data",
        num_nodes=170,
    )
    
    # Phase 3: Predictor training
    predictor_dm = PredictorDataModule(
        dataset_path="metr-imc.h5",
        train_cluster_label_path="./data/train_cluster_label.npy",
        val_cluster_label_path="./data/val_cluster_label.npy",
    )
    # ... train predictor ...
    predictor_dm.save_predicted_labels(predictor, output_dir="./data")
    
    # Phase 4: Generate sampling indices
    generate_sampling_artifacts(
        predicted_label_path="./data/predicted_label.npy",
        output_dir="./data",
    )
    
    # Phase 5: TASSGN training
    tassgn_dm = TASSGNDataModule(
        dataset_path="metr-imc.h5",
        train_sample_index_path="./data/train_sample_index.npy",
        val_sample_index_path="./data/val_sample_index.npy",
        test_sample_index_path="./data/test_sample_index.npy",
        window_data_path="./data/window_data.npy",
    )
    # ... train TASSGN ...
    ```
"""
# Datasets
from .dataset import (
    TASSGNBaseDataset,
    EncoderDataset,
    PredictorDataset,
    TASSGNDataset,
    WindowDataset,
)

# DataLoaders (collate functions)
from .dataloader import (
    encoder_collate_fn,
    predictor_collate_fn,
    tassgn_collate_fn,
    window_collate_fn,
)

# DataModules
from .datamodule import (
    TASSGNBaseDataModule,
    EncoderDataModule,
    PredictorDataModule,
    TASSGNDataModule,
)

# Phase 2: Labeler
from .labeler import (
    Labeler,
    generate_cluster_labels,
)

# Phase 4: Sampling
from .sampling import (
    generate_self_sampling_index,
    generate_train_val_test_index,
    generate_window_data,
    generate_sampling_artifacts,
    generate_window_data_from_datamodule,
)

__all__ = [
    # Datasets
    "TASSGNBaseDataset",
    "EncoderDataset",
    "PredictorDataset",
    "TASSGNDataset",
    "WindowDataset",
    # Collate functions
    "encoder_collate_fn",
    "predictor_collate_fn",
    "tassgn_collate_fn",
    "window_collate_fn",
    # DataModules
    "TASSGNBaseDataModule",
    "EncoderDataModule",
    "PredictorDataModule",
    "TASSGNDataModule",
    # Labeler (Phase 2)
    "Labeler",
    "generate_cluster_labels",
    # Sampling (Phase 4)
    "generate_self_sampling_index",
    "generate_train_val_test_index",
    "generate_window_data",
    "generate_sampling_artifacts",
    "generate_window_data_from_datamodule",
]
