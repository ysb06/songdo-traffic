import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import WandbLogger
from metr.components.adj_mx import AdjacencyMatrix
from metr.datasets.stgcn.datamodule import (
    STGCNDataModule,
    STGCNDataModuleByDate,
    STGCNSplitDataModule,
)

from metr.utils import PathConfig

from .models.stgcn.module import STGCNLightningModule
from .models.stgcn.utils import prepare_gso_for_model


def main(path_config: PathConfig):
    # Configuration
    output_dir = "./output/stgcn"

    # Data parameters
    n_his = 12  # Historical time steps
    n_pred = 3  # Prediction time steps
    batch_size = 32

    # Model parameters
    gso_type = "sym_norm_lap"  # GSO type (recommended for STGCN)
    graph_conv_type = "graph_conv"  # 'graph_conv' or 'cheb_graph_conv'
    learning_rate = 0.001

    # Training parameters
    max_epochs = 100

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("mps") if torch.backends.mps.is_available() else device
    print(f"Using device: {device}")

    # Load adjacency matrix and prepare GSO
    print("Loading adjacency matrix...")
    adj_mx_obj = AdjacencyMatrix.import_from_pickle(path_config.adj_mx_path)
    adj_mx = adj_mx_obj.adj_mx
    n_vertex = adj_mx.shape[0]
    print(f"Number of vertices (sensors): {n_vertex}")

    print(f"Preparing GSO (type: {gso_type}, graph_conv: {graph_conv_type})...")
    gso_tensor = prepare_gso_for_model(
        adj_mx=adj_mx,
        gso_type=gso_type,
        graph_conv_type=graph_conv_type,
        device=device,
        force_symmetric=True,  # Standard STGCN behavior
    )
    print(f"GSO tensor shape: {gso_tensor.shape}")

    # Initialize data module
    print("Creating data module...")
    data = STGCNSplitDataModule(
        training_data_path=path_config.metr_imc_training_path,
        test_data_path=path_config.metr_imc_test_path,
        test_missing_path=path_config.metr_imc_test_missing_path,
        adj_mx_path=path_config.adj_mx_path,
        n_his=n_his,
        n_pred=n_pred,
        batch_size=batch_size,
        num_workers=4,
        shuffle_training=True,
        train_val_split=0.8,
    )

    # Setup data module to get scaler
    print("Setting up data module...")
    data.setup("fit")
    scaler = data.scaler
    print(f"Scaler fitted: {scaler is not None}")

    # Initialize model
    print("Initializing STGCN model...")
    model = STGCNLightningModule(
        gso=gso_tensor,
        learning_rate=learning_rate,
        scheduler_factor=0.95,
        scheduler_patience=10,
        scaler=scaler,  # Pass scaler for unscaled metrics
    )

    # Setup logger and callbacks
    wandb_logger = WandbLogger(
        name="STGCN-MICE-00", project="IMC-Traffic", log_model="all"
    )

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=20,
            verbose=True,
        ),
        ModelCheckpoint(
            dirpath=output_dir,
            filename="stgcn-best-{epoch:02d}-{val_loss:.4f}",
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
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60 + "\n")
    trainer.fit(model, data)

    print("\n" + "=" * 60)
    print("Starting testing...")
    print("=" * 60 + "\n")
    trainer.test(model, data)

    print("\n" + "=" * 60)
    print("Training and testing completed!")
    print(f"Best checkpoint: {callbacks[1].best_model_path}")
    print("=" * 60)
