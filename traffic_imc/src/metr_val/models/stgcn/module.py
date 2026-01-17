import torch
import torch.nn as nn
import lightning as L
from typing import Optional, Tuple
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from argparse import Namespace
from .model import BaseSTGCN, STGCNGraphConv, STGCNChebGraphConv


class STGCNLightningModule(L.LightningModule):
    """
    PyTorch Lightning module for training STGCN model for traffic prediction
    """

    def __init__(
        self,
        gso: torch.Tensor,
        learning_rate: float = 0.001,
        scheduler_factor: float = 0.95,
        scheduler_patience: int = 10,
        scaler=None,
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
            scaler: MinMaxScaler instance for inverse transform (optional, default: None)
        """
        super().__init__()
        self.save_hyperparameters(ignore=["gso", "scaler"])
        
        # Store scaler for inverse transform (not saved in checkpoints)
        self.scaler = scaler

        # Initialize the STGCN model
        self.model = BaseSTGCN(
            n_vertex=gso.size(0),
            gso=gso,
            dropout_rate=0.5,
            n_his=12,
            Kt=3,
            stblock_num=2,
            Ks=3,
            act_func="glu",
            graph_conv_type="graph_conv",
            enable_bias=True,
        )

        # Loss function declaration
        self.criterion = nn.MSELoss()

        # Learning rate and scheduler parameters
        self.learning_rate = learning_rate
        self.scheduler_factor = scheduler_factor
        self.scheduler_patience = scheduler_patience

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
    
    def _inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Inverse transform scaled data back to original scale.
        
        Args:
            data: Scaled data of shape (batch, n_vertex)
            
        Returns:
            Unscaled data with the same shape
        """
        if self.scaler is None:
            return data
        
        original_shape = data.shape
        flat_data = data.reshape(-1, 1)
        unscaled = self.scaler.inverse_transform(flat_data)
        return unscaled.reshape(original_shape)

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Training step"""
        x, y = batch  # Simple collate function
        y_hat: torch.Tensor = self(x)

        # Reshape model output from (batch, 1, 1, n_vertex) to (batch, n_vertex)
        y_hat = y_hat.squeeze(1).squeeze(1)

        loss: torch.Tensor = self.criterion(y_hat, y)

        self.log("train_loss", loss, on_step=True, on_epoch=True)

        return loss

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Validation step"""
        x, y = batch
        y_hat: torch.Tensor = self(x)

        # Reshape model output from (batch, 1, 1, n_vertex) to (batch, n_vertex)
        y_hat = y_hat.squeeze(1).squeeze(1)

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
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Test step"""
        x, y, missing = batch  # Simple collate function
        y_hat: torch.Tensor = self(x)

        # Reshape model output from (batch, 1, 1, n_vertex) to (batch, n_vertex)
        y_hat = y_hat.squeeze(1).squeeze(1)

        loss: torch.Tensor = self.criterion(y_hat, y)

        # Store outputs for epoch-end metrics (including missing mask)
        self.test_outputs.append(
            {
                "y_true": y.cpu().numpy(),
                "y_pred": y_hat.cpu().numpy(),
                "loss": loss.item(),
                "is_missing": missing.cpu().numpy(),
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
        """Calculate test metrics at the end of epoch (excluding interpolated data)"""
        if len(self.test_outputs) == 0:
            return

        # Concatenate all predictions, targets, and missing masks
        y_true = np.concatenate([x["y_true"] for x in self.test_outputs], axis=0)
        y_pred = np.concatenate([x["y_pred"] for x in self.test_outputs], axis=0)
        is_missing = np.concatenate([x["is_missing"] for x in self.test_outputs], axis=0)

        # Flatten all arrays
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        is_missing_flat = is_missing.flatten()

        # Create mask for valid (non-interpolated) data points
        valid_mask = ~is_missing_flat  # True = original data (not interpolated)
        
        # Count statistics
        total_points = len(y_true_flat)
        valid_points = valid_mask.sum()
        interpolated_points = total_points - valid_points
        
        print(f"\nTest Data Statistics:")
        print(f"  Total points: {total_points}")
        print(f"  Valid (original) points: {valid_points} ({valid_points/total_points*100:.1f}%)")
        print(f"  Interpolated points (excluded): {interpolated_points} ({interpolated_points/total_points*100:.1f}%)")

        # Filter to only valid (non-interpolated) data
        y_true_valid = y_true_flat[valid_mask]
        y_pred_valid = y_pred_flat[valid_mask]

        # Calculate metrics on scaled data (only valid points)
        mae_scaled = mean_absolute_error(y_true_valid, y_pred_valid)
        rmse_scaled = np.sqrt(mean_squared_error(y_true_valid, y_pred_valid))
        
        # MAPE with zero-division handling
        nonzero_mask = y_true_valid != 0
        mape_scaled = (
            np.mean(np.abs((y_true_valid[nonzero_mask] - y_pred_valid[nonzero_mask]) / y_true_valid[nonzero_mask]))
            * 100
        ) if nonzero_mask.any() else 0.0

        self.log("test_mae", mae_scaled)
        self.log("test_rmse", rmse_scaled)
        self.log("test_mape", mape_scaled)

        print(f"\nTest Results (Scaled, Excluding Interpolated):")
        print(f"  MAE:  {mae_scaled:.4f}")
        print(f"  RMSE: {rmse_scaled:.4f}")
        print(f"  MAPE: {mape_scaled:.4f}%")
        
        # Calculate unscaled metrics if scaler is provided
        if self.scaler is not None:
            y_true_unscaled = self._inverse_transform(y_true)
            y_pred_unscaled = self._inverse_transform(y_pred)
            
            # Flatten and filter unscaled data
            y_true_unscaled_flat = y_true_unscaled.flatten()
            y_pred_unscaled_flat = y_pred_unscaled.flatten()
            y_true_unscaled_valid = y_true_unscaled_flat[valid_mask]
            y_pred_unscaled_valid = y_pred_unscaled_flat[valid_mask]
            
            mae_unscaled = mean_absolute_error(y_true_unscaled_valid, y_pred_unscaled_valid)
            rmse_unscaled = np.sqrt(mean_squared_error(y_true_unscaled_valid, y_pred_unscaled_valid))
            
            # MAPE with zero-division handling
            nonzero_mask_unscaled = y_true_unscaled_valid != 0
            mape_unscaled = (
                np.mean(np.abs(
                    (y_true_unscaled_valid[nonzero_mask_unscaled] - y_pred_unscaled_valid[nonzero_mask_unscaled])
                    / y_true_unscaled_valid[nonzero_mask_unscaled]
                )) * 100
            ) if nonzero_mask_unscaled.any() else 0.0
            
            self.log("test_mae_unscaled", mae_unscaled)
            self.log("test_rmse_unscaled", rmse_unscaled)
            self.log("test_mape_unscaled", mape_unscaled)
            
            print(f"\nTest Results (Original Scale, Excluding Interpolated):")
            print(f"  MAE:  {mae_unscaled:.4f}")
            print(f"  RMSE: {rmse_unscaled:.4f}")
            print(f"  MAPE: {mape_unscaled:.4f}%")

        # Clear outputs
        self.test_outputs.clear()
