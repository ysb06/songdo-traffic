"""DCRNN model training script.

Diffusion Convolutional Recurrent Neural Network for traffic prediction.
"""

import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import WandbLogger
from metr.components.adj_mx import AdjacencyMatrix
from metr.datasets.dcrnn import DCRNNSplitDataModule
from metr.utils import PathConfig

from .models.dcrnn import DCRNNLightningModule


def main(name_key: str, path_config: PathConfig, code: int = 0):
    """Main training function for DCRNN.
    
    Args:
        name_key: Name key for WandB logging (e.g., 'KNN', 'MICE')
        path_config: PathConfig instance with dataset paths
        code: Run code number for identification
    """
    # Configuration
    output_dir = "./output/dcrnn"
    
    # Data parameters
    seq_len = 12  # Historical time steps
    horizon = 12  # Prediction time steps
    batch_size = 64
    add_time_in_day = True
    add_day_in_week = False
    
    # Model parameters
    rnn_units = 64
    num_rnn_layers = 2
    max_diffusion_step = 2
    filter_type = "dual_random_walk"  # 'laplacian', 'random_walk', or 'dual_random_walk'
    use_curriculum_learning = True
    cl_decay_steps = 2000
    
    # Training parameters
    # Note: Original DCRNN uses lr=0.01 with MultiStepLR scheduler (milestones=[20,30,40,50])
    # Lower learning rate helps with convergence stability
    learning_rate = 0.001  # Changed from 0.01 - more stable for DCRNN
    weight_decay = 0.0
    max_epochs = 30
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("mps") if torch.backends.mps.is_available() else device
    print(f"Using device: {device}")
    
    # Load adjacency matrix
    print("Loading adjacency matrix...")
    adj_mx_obj = AdjacencyMatrix.import_from_pickle(path_config.adj_mx_path)
    adj_mx = adj_mx_obj.adj_mx
    
    # Initialize data module
    print("Creating data module...")
    data = DCRNNSplitDataModule(
        training_data_path=path_config.metr_imc_training_path,
        test_data_path=path_config.metr_imc_test_path,
        test_missing_path=path_config.metr_imc_test_missing_path,
        adj_mx_path=path_config.adj_mx_path,
        seq_len=seq_len,
        horizon=horizon,
        batch_size=batch_size,
        num_workers=4,
        shuffle_training=True,
        train_val_split=0.8,
        add_time_in_day=add_time_in_day,
        add_day_in_week=add_day_in_week,
    )
    
    # Setup data module to get adjacency matrix and other properties
    print("Setting up data module...")
    data.setup(stage="fit")
    
    num_nodes = data.num_nodes
    input_dim = data.input_dim
    output_dim = data.output_dim
    
    print(f"Number of nodes (sensors): {num_nodes}")
    print(f"Input dimension: {input_dim}")
    print(f"Output dimension: {output_dim}")
    print(f"Adjacency matrix shape: {adj_mx.shape}")
    
    # Initialize model
    print("Initializing DCRNN model...")
    model = DCRNNLightningModule(
        adj_mx=adj_mx,
        num_nodes=num_nodes,
        input_dim=input_dim,
        output_dim=output_dim,
        seq_len=seq_len,
        horizon=horizon,
        rnn_units=rnn_units,
        num_rnn_layers=num_rnn_layers,
        max_diffusion_step=max_diffusion_step,
        filter_type=filter_type,
        use_curriculum_learning=use_curriculum_learning,
        cl_decay_steps=cl_decay_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        scaler=data.scaler,
    )
    
    # Setup logger and callbacks
    wandb_logger = WandbLogger(
        name=f"DCRNN-{name_key}-{code:02d}", project="IMC-Traffic", log_model="all"
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
            filename="dcrnn-best-{epoch:02d}-{val_loss:.4f}",
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
        gradient_clip_val=1.0,  # Original DCRNN uses max_grad_norm=1.0
        precision="16-mixed"
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


if __name__ == "__main__":
    import os
    from .utils import parse_training_args, get_config_path
    
    args = parse_training_args()
    
    # GPU 설정
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    
    # Config 로드 및 실행
    config_path = get_config_path(args.data)
    path_config = PathConfig.from_yaml(config_path)
    main(args.data, path_config, code=args.code)
