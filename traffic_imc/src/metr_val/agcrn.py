"""
AGCRN Training Script.

Train and evaluate AGCRN (Adaptive Graph Convolutional Recurrent Network)
for traffic prediction using PyTorch Lightning.
"""
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import WandbLogger
from metr.datasets.agcrn import AGCRNDataModule

from .models.agcrn import AGCRNLightningModule


def main():
    """Main training function for AGCRN."""
    # ==========================================================================
    # Configuration
    # ==========================================================================
    dataset_path = "./data/selected_small_v1/metr-imc_train.h5"
    output_dir = "./output/agcrn"
    
    # Data parameters
    in_steps = 12       # Input time steps (lag)
    out_steps = 12      # Prediction time steps (horizon)
    batch_size = 64
    
    # Model parameters (AGCRN defaults from paper)
    input_dim = 1       # Traffic value only (no ToD/DoW)
    output_dim = 1
    rnn_units = 64      # Hidden dimension
    num_layers = 2      # Number of RNN layers
    embed_dim = 10      # Node embedding dimension
    cheb_k = 2          # Chebyshev polynomial order
    
    # Training parameters
    learning_rate = 0.003
    weight_decay = 0.0
    scheduler_step_size = 5
    scheduler_gamma = 0.7
    max_epochs = 100
    loss_func = "mae"   # 'mae' or 'mse'
    
    # ==========================================================================
    # Device setup
    # ==========================================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("mps") if torch.backends.mps.is_available() else device
    print(f"Using device: {device}")
    
    # ==========================================================================
    # Data Module
    # ==========================================================================
    print("Creating AGCRN data module...")
    data = AGCRNDataModule(
        dataset_path=dataset_path,
        training_period=("2022-11-01 00:00:00", "2024-07-31 23:59:59"),
        validation_period=("2024-08-01 00:00:00", "2024-09-30 23:59:59"),
        test_period=("2024-10-01 00:00:00", "2024-10-31 23:59:59"),
        in_steps=in_steps,
        out_steps=out_steps,
        batch_size=batch_size,
        num_workers=4,
        shuffle_training=True,
        scale_method="normal",
        normalizer="std",  # AGCRN uses StandardScaler
    )
    
    # Setup data to get num_nodes
    print("Setting up data module...")
    data.setup()
    num_nodes = data.num_nodes
    scaler = data.scaler
    print(f"Number of nodes (sensors): {num_nodes}")
    print(f"Train samples: {len(data.train_dataset)}")
    print(f"Val samples: {len(data.val_dataset)}")
    print(f"Test samples: {len(data.test_dataset)}")
    
    # ==========================================================================
    # Model
    # ==========================================================================
    print("\nInitializing AGCRN model...")
    model = AGCRNLightningModule(
        num_nodes=num_nodes,
        in_steps=in_steps,
        out_steps=out_steps,
        input_dim=input_dim,
        output_dim=output_dim,
        rnn_units=rnn_units,
        num_layers=num_layers,
        embed_dim=embed_dim,
        cheb_k=cheb_k,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        scheduler_step_size=scheduler_step_size,
        scheduler_gamma=scheduler_gamma,
        loss_func=loss_func,
        scaler=scaler,
    )
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # ==========================================================================
    # Logger and Callbacks
    # ==========================================================================
    wandb_logger = WandbLogger(
        project="Traffic-IMC-AGCRN",
        log_model="all",
        save_dir=output_dir,
    )
    
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=15,
            verbose=True,
        ),
        ModelCheckpoint(
            dirpath=output_dir,
            filename="agcrn-best-{epoch:02d}-{val_loss:.4f}",
            save_top_k=3,
            monitor="val_loss",
            mode="min",
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]
    
    # ==========================================================================
    # Trainer
    # ==========================================================================
    trainer = Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices="auto",
        default_root_dir=output_dir,
        logger=wandb_logger,
        callbacks=callbacks,
        log_every_n_steps=10,
        enable_progress_bar=True,
        gradient_clip_val=5.0,  # AGCRN uses gradient clipping
    )
    
    # ==========================================================================
    # Training
    # ==========================================================================
    print("\n" + "=" * 60)
    print("Starting AGCRN training...")
    print("=" * 60 + "\n")
    trainer.fit(model, data)
    
    # ==========================================================================
    # Testing
    # ==========================================================================
    print("\n" + "=" * 60)
    print("Starting testing...")
    print("=" * 60 + "\n")
    trainer.test(model, data)
    
    print("\n" + "=" * 60)
    print("Training and testing completed!")
    print(f"Best checkpoint: {callbacks[1].best_model_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
