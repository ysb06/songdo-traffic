import torch
import torch.nn as nn
import lightning as L
from typing import Optional, Tuple
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from argparse import Namespace
from .model import STGCNGraphConv


class STGCNLightningModule(L.LightningModule):
    """
    PyTorch Lightning module for training STGCN model for traffic prediction
    """

    def __init__(
        self,
        n_vertex: int,
        gso: torch.Tensor,
        input_size: int = 1,
        hidden_size: int = 64,
        num_layers: int = 2,
        output_size: int = 1,
        learning_rate: float = 0.001,
        dropout_rate: float = 0.5,
        scheduler_factor: float = 0.95,
        scheduler_patience: int = 10,
        n_his: int = 12,
        n_pred: int = 3,
        Kt: int = 3,
        stblock_num: int = 2,
        Ks: int = 3,
        act_func: str = "glu",
        graph_conv_type: str = "graph_conv",
        enable_bias: bool = True,
    ):
        """
        Args:
            n_vertex: Number of nodes in the graph
            gso: Graph shift operator (adjacency matrix or Laplacian)
            input_size: Input feature size (default: 1)
            hidden_size: Hidden layer size (default: 64)
            num_layers: Number of layers (default: 2)
            output_size: Output feature size (default: 1)
            learning_rate: Learning rate (default: 0.001)
            dropout_rate: Dropout rate (default: 0.5)
            scheduler_factor: LR scheduler factor (default: 0.95)
            scheduler_patience: LR scheduler patience (default: 10)
            n_his: Historical time steps (default: 12)
            n_pred: Prediction time steps (default: 3)
            Kt: Kernel size of temporal convolution (default: 3)
            stblock_num: Number of ST-Conv blocks (default: 2)
            Ks: Kernel size of spatial convolution (default: 3)
            act_func: Activation function ('glu' or 'gtu', default: 'glu')
            graph_conv_type: Graph convolution type ('cheb_graph_conv' or 'graph_conv', default: 'graph_conv')
            enable_bias: Enable bias in layers (default: True)
        """
        super().__init__()
        self.save_hyperparameters(ignore=["gso"])

        # Calculate Ko (output temporal dimension)
        Ko = n_his - (Kt - 1) * 2 * stblock_num

        # Build blocks configuration
        # blocks: settings of channel size in st_conv_blocks and output layer,
        # using the bottleneck design in st_conv_blocks
        blocks = []
        blocks.append([1])
        for l in range(stblock_num):
            blocks.append([64, 16, 64])
        if Ko == 0:
            blocks.append([128])
        elif Ko > 0:
            blocks.append([128, 128])
        blocks.append([1])

        # Create args namespace for STGCN model
        args = Namespace(
            n_his=n_his,
            Kt=Kt,
            Ks=Ks,
            act_func=act_func,
            graph_conv_type=graph_conv_type,
            gso=gso,
            enable_bias=enable_bias,
            droprate=dropout_rate,
        )

        # Initialize the STGCN model
        self.model = STGCNGraphConv(args, blocks, n_vertex)

        # Loss function declaration
        self.criterion = nn.MSELoss()

        # Learning rate and scheduler parameters
        self.learning_rate = learning_rate
        self.scheduler_factor = scheduler_factor
        self.scheduler_patience = scheduler_patience

        # Store additional parameters
        self.n_pred = n_pred

        # Metrics storage
        self.validation_outputs = []
        self.test_outputs = []

    def configure_optimizers(self):
        """Configure optimizer with StepLR scheduler"""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.scheduler_patience,
            gamma=self.scheduler_factor,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Training step"""
        x, y = batch  # Simple collate function
        y_hat: torch.Tensor = self(x)

        # Reshape y if needed
        if y.dim() > 2:
            y = y.squeeze(-1)  # Remove last dimension if it's 1

        loss: torch.Tensor = self.criterion(y_hat, y)

        self.log("train_loss", loss, on_step=True, on_epoch=True)

        return loss

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Validation step"""
        x, y = batch
        y_hat: torch.Tensor = self(x)

        # Reshape y if needed
        if y.dim() > 2:
            y = y.squeeze(-1)

        loss: torch.Tensor = self.criterion(y_hat, y)

        # Store outputs for epoch-end metrics
        self.validation_outputs.append(
            {
                "y_true": y.cpu().numpy(),
                "y_pred": y_hat.cpu().numpy(),
                "loss": loss.item(),
            }
        )

        self.log("val_loss", loss, on_step=False, on_epoch=True)

        return loss

    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Test step"""
        x, y = batch  # Simple collate function
        y_hat: torch.Tensor = self(x)

        # Reshape y if needed
        if y.dim() > 2:
            y = y.squeeze(-1)

        loss: torch.Tensor = self.criterion(y_hat, y)

        # Store outputs for epoch-end metrics
        self.test_outputs.append(
            {
                "y_true": y.cpu().numpy(),
                "y_pred": y_hat.cpu().numpy(),
                "loss": loss.item(),
            }
        )

        self.log("test_loss", loss, on_step=False, on_epoch=True)

        return loss

    def on_validation_epoch_end(self):
        """Calculate validation metrics at the end of epoch"""
        if len(self.validation_outputs) == 0:
            return

        # Concatenate all predictions and targets
        y_true = np.concatenate([x["y_true"] for x in self.validation_outputs], axis=0)
        y_pred = np.concatenate([x["y_pred"] for x in self.validation_outputs], axis=0)

        # Calculate metrics on scaled data
        mae_scaled = mean_absolute_error(y_true.flatten(), y_pred.flatten())
        rmse_scaled = np.sqrt(mean_squared_error(y_true.flatten(), y_pred.flatten()))

        self.log("val_mae", mae_scaled)
        self.log("val_rmse", rmse_scaled)

        # Clear outputs for next epoch
        self.validation_outputs.clear()

    def on_test_epoch_end(self):
        """Calculate test metrics at the end of epoch"""
        if len(self.test_outputs) == 0:
            return

        # Concatenate all predictions and targets
        y_true = np.concatenate([x["y_true"] for x in self.test_outputs], axis=0)
        y_pred = np.concatenate([x["y_pred"] for x in self.test_outputs], axis=0)

        # Calculate metrics on scaled data
        mae_scaled = mean_absolute_error(y_true.flatten(), y_pred.flatten())
        rmse_scaled = np.sqrt(mean_squared_error(y_true.flatten(), y_pred.flatten()))
        mape_scaled = (
            np.mean(np.abs((y_true.flatten() - y_pred.flatten()) / y_true.flatten()))
            * 100
        )

        self.log("test_mae", mae_scaled)
        self.log("test_rmse", rmse_scaled)
        self.log("test_mape", mape_scaled)

        print(f"\nTest Results (Scaled):")
        print(f"MAE: {mae_scaled:.4f}")
        print(f"RMSE: {rmse_scaled:.4f}")
        print(f"MAPE: {mape_scaled:.4f}%")

        # Clear outputs
        self.test_outputs.clear()
