"""
PyTorch Lightning Module for AGCRN.

Provides training, validation, and testing loops with:
- MSE/MAE loss functions
- Learning rate scheduling
- Metrics logging (MAE, RMSE, MAPE)
- Support for scaled/unscaled evaluation
"""
from typing import Optional, Tuple

import lightning as L
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .agcrn import AGCRN


class AGCRNLightningModule(L.LightningModule):
    """PyTorch Lightning module for training AGCRN model.
    
    Args:
        num_nodes: Number of nodes in the graph
        in_steps: Number of input time steps (default: 12)
        out_steps: Number of output time steps (default: 12)
        input_dim: Input feature dimension (default: 1)
        output_dim: Output feature dimension (default: 1)
        rnn_units: Hidden dimension of RNN (default: 64)
        num_layers: Number of RNN layers (default: 2)
        embed_dim: Node embedding dimension (default: 10)
        cheb_k: Order of Chebyshev polynomials (default: 2)
        learning_rate: Learning rate (default: 0.003)
        weight_decay: Weight decay for optimizer (default: 0)
        scheduler_step_size: LR scheduler step size (default: 5)
        scheduler_gamma: LR scheduler gamma (default: 0.7)
        loss_func: Loss function - 'mse' or 'mae' (default: 'mae')
        scaler: Scaler instance for inverse transform (optional)
    """
    
    def __init__(
        self,
        num_nodes: int,
        in_steps: int = 12,
        out_steps: int = 12,
        input_dim: int = 1,
        output_dim: int = 1,
        rnn_units: int = 64,
        num_layers: int = 2,
        embed_dim: int = 10,
        cheb_k: int = 2,
        learning_rate: float = 0.003,
        weight_decay: float = 0.0,
        scheduler_step_size: int = 5,
        scheduler_gamma: float = 0.7,
        loss_func: str = "mae",
        scaler=None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["scaler"])
        
        # Store scaler for inverse transform
        self.scaler = scaler
        
        # Model parameters
        self.num_nodes = num_nodes
        self.in_steps = in_steps
        self.out_steps = out_steps
        
        # Initialize AGCRN model
        self.model = AGCRN(
            num_nodes=num_nodes,
            input_dim=input_dim,
            output_dim=output_dim,
            horizon=out_steps,
            rnn_units=rnn_units,
            num_layers=num_layers,
            embed_dim=embed_dim,
            cheb_k=cheb_k,
        )
        
        # Loss function
        if loss_func == "mse":
            self.criterion = nn.MSELoss()
        else:  # mae
            self.criterion = nn.L1Loss()
        
        # Optimizer parameters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.scheduler_step_size = scheduler_step_size
        self.scheduler_gamma = scheduler_gamma
        
        # Metrics storage
        self.validation_outputs = []
        self.test_outputs = []
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (B, T, N, D)
            
        Returns:
            Predictions of shape (B, horizon, N, output_dim)
        """
        return self.model(x)
    
    def _inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Inverse transform scaled data back to original scale.
        
        Args:
            data: Scaled data of shape (batch, out_steps, num_nodes)
            
        Returns:
            Unscaled data with the same shape
        """
        if self.scaler is None:
            return data
        
        original_shape = data.shape
        flat_data = data.reshape(-1, 1)
        unscaled = self.scaler.inverse_transform(flat_data)
        return unscaled.reshape(original_shape)
    
    def configure_optimizers(self):
        """Configure optimizer with StepLR scheduler."""
        optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.scheduler_step_size,
            gamma=self.scheduler_gamma,
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }
    
    def training_step(
        self, 
        batch: Tuple[torch.Tensor, torch.Tensor], 
        batch_idx: int
    ) -> torch.Tensor:
        """Training step.
        
        Args:
            batch: Tuple of (x, y) tensors
            batch_idx: Batch index
            
        Returns:
            Loss tensor
        """
        x, y = batch
        
        # Forward pass
        y_hat = self(x)  # (B, horizon, N, output_dim)
        
        # Squeeze output_dim if it's 1
        if y_hat.dim() == 4 and y_hat.size(-1) == 1:
            y_hat = y_hat.squeeze(-1)  # (B, horizon, N)
        
        if y.dim() == 4 and y.size(-1) == 1:
            y = y.squeeze(-1)
        
        loss = self.criterion(y_hat, y)
        
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(
        self, 
        batch: Tuple[torch.Tensor, torch.Tensor], 
        batch_idx: int
    ) -> torch.Tensor:
        """Validation step.
        
        Args:
            batch: Tuple of (x, y) tensors
            batch_idx: Batch index
            
        Returns:
            Loss tensor
        """
        x, y = batch
        
        y_hat = self(x)
        
        if y_hat.dim() == 4 and y_hat.size(-1) == 1:
            y_hat = y_hat.squeeze(-1)
        
        if y.dim() == 4 and y.size(-1) == 1:
            y = y.squeeze(-1)
        
        loss = self.criterion(y_hat, y)
        
        # Store outputs for epoch-end metrics
        self.validation_outputs.append({
            "y_true": y.cpu().numpy(),
            "y_pred": y_hat.cpu().numpy(),
            "loss": loss.item(),
        })
        
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def test_step(
        self, 
        batch: Tuple[torch.Tensor, torch.Tensor], 
        batch_idx: int
    ) -> torch.Tensor:
        """Test step.
        
        Args:
            batch: Tuple of (x, y) tensors
            batch_idx: Batch index
            
        Returns:
            Loss tensor
        """
        x, y = batch
        
        y_hat = self(x)
        
        if y_hat.dim() == 4 and y_hat.size(-1) == 1:
            y_hat = y_hat.squeeze(-1)
        
        if y.dim() == 4 and y.size(-1) == 1:
            y = y.squeeze(-1)
        
        loss = self.criterion(y_hat, y)
        
        # Store outputs for epoch-end metrics
        self.test_outputs.append({
            "y_true": y.cpu().numpy(),
            "y_pred": y_hat.cpu().numpy(),
            "loss": loss.item(),
        })
        
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        
        return loss
    
    def on_validation_epoch_end(self):
        """Calculate validation metrics at the end of epoch."""
        if len(self.validation_outputs) == 0:
            return
        
        # Concatenate all predictions and targets
        y_true = np.concatenate([x["y_true"] for x in self.validation_outputs], axis=0)
        y_pred = np.concatenate([x["y_pred"] for x in self.validation_outputs], axis=0)
        
        # Calculate metrics on scaled data
        mae_scaled = mean_absolute_error(y_true.flatten(), y_pred.flatten())
        rmse_scaled = np.sqrt(mean_squared_error(y_true.flatten(), y_pred.flatten()))
        
        self.log("val_mae", mae_scaled, prog_bar=True)
        self.log("val_rmse", rmse_scaled)
        
        # Clear outputs for next epoch
        self.validation_outputs.clear()
    
    def on_test_epoch_end(self):
        """Calculate test metrics at the end of epoch."""
        if len(self.test_outputs) == 0:
            return
        
        # Concatenate all predictions and targets
        y_true = np.concatenate([x["y_true"] for x in self.test_outputs], axis=0)
        y_pred = np.concatenate([x["y_pred"] for x in self.test_outputs], axis=0)
        
        # Calculate metrics on scaled data
        mae_scaled = mean_absolute_error(y_true.flatten(), y_pred.flatten())
        rmse_scaled = np.sqrt(mean_squared_error(y_true.flatten(), y_pred.flatten()))
        
        # MAPE with zero-division handling
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        mask = y_true_flat != 0
        mape_scaled = (
            np.mean(np.abs((y_true_flat[mask] - y_pred_flat[mask]) / y_true_flat[mask]))
            * 100
        ) if mask.any() else 0.0
        
        self.log("test_mae", mae_scaled)
        self.log("test_rmse", rmse_scaled)
        self.log("test_mape", mape_scaled)
        
        print(f"\nTest Results (Scaled):")
        print(f"  MAE:  {mae_scaled:.4f}")
        print(f"  RMSE: {rmse_scaled:.4f}")
        print(f"  MAPE: {mape_scaled:.4f}%")
        
        # Calculate unscaled metrics if scaler is provided
        if self.scaler is not None:
            y_true_unscaled = self._inverse_transform(y_true)
            y_pred_unscaled = self._inverse_transform(y_pred)
            
            mae_unscaled = mean_absolute_error(
                y_true_unscaled.flatten(), y_pred_unscaled.flatten()
            )
            rmse_unscaled = np.sqrt(mean_squared_error(
                y_true_unscaled.flatten(), y_pred_unscaled.flatten()
            ))
            
            # MAPE with zero-division handling
            y_true_unscaled_flat = y_true_unscaled.flatten()
            y_pred_unscaled_flat = y_pred_unscaled.flatten()
            mask_unscaled = y_true_unscaled_flat != 0
            mape_unscaled = (
                np.mean(np.abs(
                    (y_true_unscaled_flat[mask_unscaled] - y_pred_unscaled_flat[mask_unscaled]) 
                    / y_true_unscaled_flat[mask_unscaled]
                )) * 100
            ) if mask_unscaled.any() else 0.0
            
            self.log("test_mae_unscaled", mae_unscaled)
            self.log("test_rmse_unscaled", rmse_unscaled)
            self.log("test_mape_unscaled", mape_unscaled)
            
            print(f"\nTest Results (Original Scale):")
            print(f"  MAE:  {mae_unscaled:.4f}")
            print(f"  RMSE: {rmse_unscaled:.4f}")
            print(f"  MAPE: {mape_unscaled:.4f}%")
        
        # Clear outputs
        self.test_outputs.clear()
