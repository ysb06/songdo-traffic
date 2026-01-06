"""
TASSGN 5-Phase Training Pipeline.

Phase 1: STIDEncoder pretraining (self-supervised reconstruction)
Phase 2: Cluster label generation (using trained encoder)
Phase 3: Predictor training (label prediction)
Phase 4: Self-sampling index generation (using trained predictor)
Phase 5: TASSGN final training (traffic prediction)
"""

import os
from pathlib import Path

import numpy as np
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import WandbLogger

from metr.datasets.tassgn import (
    EncoderDataModule,
    PredictorDataModule,
    TASSGNDataModule,
    Labeler,
    generate_self_sampling_index,
    generate_window_data_from_datamodule,
)

from .models.tassgn import (
    STIDEncoderLightningModule,
    PredictorLightningModule,
    TASSGNLightningModule,
)


def main():
    """Run the complete TASSGN 5-phase training pipeline."""
    
    # ==================== Configuration ====================
    # Data paths
    dataset_path = "./data/selected_small_v1/metr-imc_train.h5"
    test_dataset_path = "./data/selected_small_v1/metr-imc_test.h5"
    output_dir = Path("./output/tassgn")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Artifact paths (for intermediate files)
    train_repr_path = output_dir / "train_representation.npy"
    val_repr_path = output_dir / "val_representation.npy"
    train_cluster_label_path = output_dir / "train_cluster_label.npy"
    val_cluster_label_path = output_dir / "val_cluster_label.npy"
    predicted_label_path = output_dir / "predicted_label.npy"
    train_sample_index_path = output_dir / "train_sample_index.npy"
    val_sample_index_path = output_dir / "val_sample_index.npy"
    test_sample_index_path = output_dir / "test_sample_index.npy"
    window_data_path = output_dir / "window_data.npy"
    
    # Data parameters
    in_steps = 12
    out_steps = 12
    steps_per_day = 288
    batch_size = 32
    
    # Model parameters (from TASSGN paper)
    input_dim = 1  # Traffic flow only (temporal features handled separately)
    hid_dim = 32
    num_layers = 3
    num_clusters = 20
    num_samples = 7
    num_blocks = 2
    num_attention_heads = 2
    topk = 10
    dropout = 0.1
    mask_ratio = 0.5
    learning_rate = 0.002
    time_of_day_size = 288
    day_of_week_size = 7
    
    # Labeler parameters
    cluster_thresh = 30
    cluster_repeat = 10
    
    # Training parameters
    max_epochs_encoder = 50
    max_epochs_predictor = 50
    max_epochs_tassgn = 100
    
    # Periods
    training_period = ("2022-11-01 00:00:00", "2024-07-31 23:59:59")
    validation_period = ("2024-08-01 00:00:00", "2024-09-30 23:59:59")
    test_period = ("2024-10-01 00:00:00", "2024-10-31 23:59:59")
    
    # Device info
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("mps") if torch.backends.mps.is_available() else device
    print(f"Using device: {device}")
    
    # ==================== Phase 1: Encoder Pretraining ====================
    print("\n" + "=" * 60)
    print("PHASE 1: STIDEncoder Pretraining (Self-supervised)")
    print("=" * 60 + "\n")
    
    # Initialize encoder data module
    encoder_data = EncoderDataModule(
        dataset_path=dataset_path,
        training_period=training_period,
        validation_period=validation_period,
        test_period=test_period,
        out_steps=out_steps,
        steps_per_day=steps_per_day,
        batch_size=batch_size,
        num_workers=4,
        shuffle_training=True,
    )
    encoder_data.setup()
    
    assert encoder_data.num_nodes is not None
    num_nodes = encoder_data.num_nodes
    scaler = encoder_data.scaler
    print(f"Number of nodes (sensors): {num_nodes}")
    
    # Initialize encoder model
    encoder_model = STIDEncoderLightningModule(
        num_nodes=num_nodes,
        input_len=out_steps,  # Encoder uses y_data (out_steps)
        output_len=out_steps,
        input_dim=input_dim,
        hid_dim=hid_dim,
        num_layers=num_layers,
        time_of_day_size=time_of_day_size,
        day_of_week_size=day_of_week_size,
        mask_ratio=mask_ratio,
        learning_rate=learning_rate,
    )
    
    # Setup logger and callbacks for Phase 1
    wandb_logger_encoder = WandbLogger(
        project="Traffic-IMC-TASSGN",
        name="phase1-encoder",
        log_model="all",
    )
    
    encoder_callbacks = [
        EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=10,
            verbose=True,
        ),
        ModelCheckpoint(
            dirpath=str(output_dir),
            filename="encoder-best-{epoch:02d}-{val_loss:.4f}",
            save_top_k=1,
            monitor="val_loss",
            mode="min",
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]
    
    encoder_trainer = Trainer(
        max_epochs=max_epochs_encoder,
        accelerator="auto",
        devices="auto",
        default_root_dir=str(output_dir),
        logger=wandb_logger_encoder,
        callbacks=encoder_callbacks,
        log_every_n_steps=10,
        enable_progress_bar=True,
    )
    
    # Train encoder
    encoder_trainer.fit(encoder_model, encoder_data)
    
    # Finish Phase 1 wandb run
    wandb_logger_encoder.experiment.finish()
    
    # ==================== Phase 2: Cluster Label Generation ====================
    print("\n" + "=" * 60)
    print("PHASE 2: Cluster Label Generation")
    print("=" * 60 + "\n")
    
    # Generate and save representations using DataModule method
    print("Generating representations...")
    train_repr, val_repr = encoder_data.save_representations(
        encoder=encoder_model.model,
        output_dir=str(output_dir),
        device=str(device),
    )
    
    # Generate cluster labels using Labeler
    print("Generating cluster labels...")
    labeler = Labeler(
        num_nodes=num_nodes,
        thresh=cluster_thresh,
        repeat=cluster_repeat,
    )
    train_labels, val_labels = labeler.generate_labels(train_repr, val_repr)
    
    # Save cluster labels
    np.save(train_cluster_label_path, train_labels)
    np.save(val_cluster_label_path, val_labels)
    print(f"Train cluster labels saved: {train_labels.shape}")
    print(f"Val cluster labels saved: {val_labels.shape}")
    
    # ==================== Phase 3: Predictor Training ====================
    print("\n" + "=" * 60)
    print("PHASE 3: Predictor Training (Label Classification)")
    print("=" * 60 + "\n")
    
    # Initialize predictor data module with cluster labels
    predictor_data = PredictorDataModule(
        dataset_path=dataset_path,
        train_cluster_label_path=str(train_cluster_label_path),
        val_cluster_label_path=str(val_cluster_label_path),
        training_period=training_period,
        validation_period=validation_period,
        test_period=test_period,
        in_steps=in_steps,
        out_steps=out_steps,
        steps_per_day=steps_per_day,
        batch_size=batch_size,
        num_workers=4,
        shuffle_training=True,
    )
    predictor_data.setup()
    
    assert predictor_data.num_clusters is not None
    print(f"Number of clusters: {predictor_data.num_clusters}")
    
    # Get encoder state dict for predictor initialization
    encoder_state_dict = encoder_model.get_encoder_state_dict()
    
    # Initialize predictor model
    predictor_model = PredictorLightningModule(
        num_nodes=num_nodes,
        input_len=in_steps,
        input_dim=input_dim,
        hid_dim=hid_dim,
        num_clusters=predictor_data.num_clusters,
        num_layers=num_layers,
        time_of_day_size=time_of_day_size,
        day_of_week_size=day_of_week_size,
        encoder_state_dict=encoder_state_dict,
        learning_rate=learning_rate,
    )
    
    # Setup logger and callbacks for Phase 3
    wandb_logger_predictor = WandbLogger(
        project="Traffic-IMC-TASSGN",
        name="phase3-predictor",
        log_model="all",
    )
    
    predictor_callbacks = [
        EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=10,
            verbose=True,
        ),
        ModelCheckpoint(
            dirpath=str(output_dir),
            filename="predictor-best-{epoch:02d}-{val_loss:.4f}",
            save_top_k=1,
            monitor="val_loss",
            mode="min",
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]
    
    predictor_trainer = Trainer(
        max_epochs=max_epochs_predictor,
        accelerator="auto",
        devices="auto",
        default_root_dir=str(output_dir),
        logger=wandb_logger_predictor,
        callbacks=predictor_callbacks,
        log_every_n_steps=10,
        enable_progress_bar=True,
    )
    
    # Train predictor
    predictor_trainer.fit(predictor_model, predictor_data)
    
    # Finish Phase 3 wandb run
    wandb_logger_predictor.experiment.finish()
    
    # ==================== Phase 4: Self-Sampling Index Generation ====================
    print("\n" + "=" * 60)
    print("PHASE 4: Self-Sampling Index Generation")
    print("=" * 60 + "\n")
    
    # Generate predicted labels using DataModule method
    print("Generating predicted labels...")
    pred_labels = predictor_data.save_predicted_labels(
        predictor=predictor_model.model,
        output_dir=str(output_dir),
        device=str(device),
    )
    
    # Generate self-sampling indices
    print("Generating self-sampling indices...")
    sample_index = generate_self_sampling_index(
        pred_label=pred_labels,
        num_samples=num_samples,
        history_len=in_steps,
        future_len=out_steps,
    )
    
    # Split into train/val/test based on periods
    # Calculate split sizes based on data periods
    total_samples = sample_index.shape[0]
    train_size = len(encoder_data.train_dataset) if encoder_data.train_dataset else 0
    val_size = len(encoder_data.val_dataset) if encoder_data.val_dataset else 0
    test_size = total_samples - train_size - val_size
    
    print(f"Splitting sample indices - Train: {train_size}, Val: {val_size}, Test: {test_size}")
    
    # Adjust for actual sample_index length
    if train_size + val_size + test_size > total_samples:
        # Recalculate proportionally
        train_ratio = train_size / (train_size + val_size + test_size)
        val_ratio = val_size / (train_size + val_size + test_size)
        
        train_size = int(total_samples * train_ratio)
        val_size = int(total_samples * val_ratio)
        test_size = total_samples - train_size - val_size
    
    train_sample_index = sample_index[:train_size]
    val_sample_index = sample_index[train_size:train_size + val_size]
    test_sample_index = sample_index[train_size + val_size:]
    
    # Save sample indices
    np.save(train_sample_index_path, train_sample_index)
    np.save(val_sample_index_path, val_sample_index)
    np.save(test_sample_index_path, test_sample_index)
    
    print(f"Train sample index saved: {train_sample_index.shape}")
    print(f"Val sample index saved: {val_sample_index.shape}")
    print(f"Test sample index saved: {test_sample_index.shape}")
    
    # Generate window data for self-sampling
    print("Generating window data...")
    window_data = generate_window_data_from_datamodule(
        dataset_path=dataset_path,
        output_dir=str(output_dir),
        future_len=out_steps,
        steps_per_day=steps_per_day,
        scaler=scaler,
    )
    print(f"Window data saved: {window_data.shape}")
    
    # ==================== Phase 5: TASSGN Final Training ====================
    print("\n" + "=" * 60)
    print("PHASE 5: TASSGN Final Training (Traffic Prediction)")
    print("=" * 60 + "\n")
    
    # Initialize TASSGN data module with sample indices
    tassgn_data = TASSGNDataModule(
        dataset_path=dataset_path,
        train_sample_index_path=str(train_sample_index_path),
        val_sample_index_path=str(val_sample_index_path),
        test_sample_index_path=str(test_sample_index_path),
        window_data_path=str(window_data_path),
        training_period=training_period,
        validation_period=validation_period,
        test_period=test_period,
        in_steps=in_steps,
        out_steps=out_steps,
        steps_per_day=steps_per_day,
        num_samples=num_samples,
        batch_size=batch_size,
        num_workers=4,
        shuffle_training=True,
    )
    tassgn_data.setup()
    
    # Initialize TASSGN model
    tassgn_model = TASSGNLightningModule(
        num_nodes=num_nodes,
        input_len=in_steps,
        output_len=out_steps,
        input_dim=input_dim,
        hid_dim=hid_dim,
        num_samples=num_samples,
        num_layers=num_layers,
        num_blocks=num_blocks,
        num_attention_heads=num_attention_heads,
        topk=topk,
        dropout=dropout,
        time_of_day_size=time_of_day_size,
        day_of_week_size=day_of_week_size,
        learning_rate=learning_rate,
        scaler=scaler,  # For unscaled metrics
    )
    
    # Setup logger and callbacks for Phase 5
    wandb_logger_tassgn = WandbLogger(
        project="Traffic-IMC-TASSGN",
        name="phase5-tassgn",
        log_model="all",
    )
    
    tassgn_callbacks = [
        EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=20,
            verbose=True,
        ),
        ModelCheckpoint(
            dirpath=str(output_dir),
            filename="tassgn-best-{epoch:02d}-{val_loss:.4f}",
            save_top_k=3,
            monitor="val_loss",
            mode="min",
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]
    
    tassgn_trainer = Trainer(
        max_epochs=max_epochs_tassgn,
        accelerator="auto",
        devices="auto",
        default_root_dir=str(output_dir),
        logger=wandb_logger_tassgn,
        callbacks=tassgn_callbacks,
        log_every_n_steps=10,
        enable_progress_bar=True,
    )
    
    # Train TASSGN
    tassgn_trainer.fit(tassgn_model, tassgn_data)
    
    # Test TASSGN
    print("\n" + "=" * 60)
    print("Testing TASSGN Model...")
    print("=" * 60 + "\n")
    tassgn_trainer.test(tassgn_model, tassgn_data)
    
    # Finish Phase 5 wandb run
    wandb_logger_tassgn.experiment.finish()
    
    # ==================== Training Complete ====================
    print("\n" + "=" * 60)
    print("TASSGN 5-Phase Training Pipeline Completed!")
    print("=" * 60)
    print(f"\nArtifacts saved to: {output_dir}")
    print(f"  - Train representation: {train_repr_path}")
    print(f"  - Val representation: {val_repr_path}")
    print(f"  - Train cluster labels: {train_cluster_label_path}")
    print(f"  - Val cluster labels: {val_cluster_label_path}")
    print(f"  - Predicted labels: {predicted_label_path}")
    print(f"  - Train sample index: {train_sample_index_path}")
    print(f"  - Val sample index: {val_sample_index_path}")
    print(f"  - Test sample index: {test_sample_index_path}")
    print(f"  - Window data: {window_data_path}")
    print(f"  - TASSGN best checkpoint: {tassgn_callbacks[1].best_model_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
