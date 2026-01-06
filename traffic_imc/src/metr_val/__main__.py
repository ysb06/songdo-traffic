import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import WandbLogger
from metr.datasets.mlcaformer import MLCAFormerDataModule

from .models.mlcaformer import MLCAFormerLightningModule


def main():
    # Configuration
    dataset_path = "./data/selected_small_v1/metr-imc_train.h5"
    test_dataset_path = "./data/selected_small_v1/metr-imc_test.h5"
    output_dir = "./output/mlcaformer"
    
    # Data parameters
    in_steps = 12   # Input time steps
    out_steps = 12  # Prediction time steps
    steps_per_day = 288  # 5-minute intervals (24 * 60 / 5 = 288)
    batch_size = 32
    
    # Model parameters
    input_dim = 3  # traffic_value, time_of_day, day_of_week
    output_dim = 1
    input_embedding_dim = 24
    tod_embedding_dim = 24
    dow_embedding_dim = 24
    nid_embedding_dim = 24
    col_embedding_dim = 80
    feed_forward_dim = 256
    num_heads = 4
    num_layers = 3
    dropout = 0.1
    learning_rate = 0.001
    
    # Training parameters
    max_epochs = 100
    
    # Device info
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("mps") if torch.backends.mps.is_available() else device
    print(f"Using device: {device}")
    
    # Initialize data module
    print("Creating MLCAFormer data module...")
    data = MLCAFormerDataModule(
        dataset_path=dataset_path,
        training_period=("2022-11-01 00:00:00", "2024-07-31 23:59:59"),
        validation_period=("2024-08-01 00:00:00", "2024-09-30 23:59:59"),
        test_period=("2024-10-01 00:00:00", "2024-10-31 23:59:59"),
        in_steps=in_steps,
        out_steps=out_steps,
        steps_per_day=steps_per_day,
        batch_size=batch_size,
        num_workers=4,
        shuffle_training=True,
        scale_method="normal",
    )
    
    # Setup data to get num_nodes
    print("Setting up data module...")
    data.setup()
    num_nodes = data.num_nodes
    scaler = data.scaler  # Get scaler for inverse transform
    print(f"Number of nodes (sensors): {num_nodes}")
    
    # Initialize model
    print("Initializing MLCAFormer model...")
    model = MLCAFormerLightningModule(
        num_nodes=num_nodes,
        in_steps=in_steps,
        out_steps=out_steps,
        steps_per_day=steps_per_day,
        input_dim=input_dim,
        output_dim=output_dim,
        input_embedding_dim=input_embedding_dim,
        tod_embedding_dim=tod_embedding_dim,
        dow_embedding_dim=dow_embedding_dim,
        nid_embedding_dim=nid_embedding_dim,
        col_embedding_dim=col_embedding_dim,
        feed_forward_dim=feed_forward_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
        learning_rate=learning_rate,
        scheduler_factor=0.5,
        scheduler_patience=10,
        scaler=scaler,  # Pass scaler for unscaled metrics
    )
    
    # Setup logger and callbacks
    wandb_logger = WandbLogger(project="Traffic-IMC-MLCAFormer", log_model="all")
    
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=20,
            verbose=True,
        ),
        ModelCheckpoint(
            dirpath=output_dir,
            filename="mlcaformer-best-{epoch:02d}-{val_loss:.4f}",
            save_top_k=3,
            monitor="val_loss",
            mode="min",
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]
    
    # Initialize trainer
    trainer = Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices="auto",
        default_root_dir=output_dir,
        logger=wandb_logger,
        callbacks=callbacks,
        log_every_n_steps=10,
        enable_progress_bar=True,
    )
    
    # Train and test
    print("\n" + "="*60)
    print("Starting MLCAFormer training...")
    print("="*60 + "\n")
    trainer.fit(model, data)
    
    print("\n" + "="*60)
    print("Starting testing...")
    print("="*60 + "\n")
    trainer.test(model, data)
    
    print("\n" + "="*60)
    print("Training and testing completed!")
    print(f"Best checkpoint: {callbacks[1].best_model_path}")
    print("="*60)


if __name__ == "__main__":
    main()
