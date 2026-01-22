from typing import Tuple

import lightning as L
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .mlcaformer import MLCAFormer


class MLCAFormerLightningModule(L.LightningModule):
    """
    PyTorch Lightning module for training MLCAFormer model for traffic prediction.
    """

    def __init__(
        self,
        num_nodes: int,
        in_steps: int = 12,
        out_steps: int = 12,
        steps_per_day: int = 288,
        input_dim: int = 3,
        output_dim: int = 1,
        input_embedding_dim: int = 24,
        tod_embedding_dim: int = 24,
        dow_embedding_dim: int = 24,
        nid_embedding_dim: int = 24,
        col_embedding_dim: int = 80,
        feed_forward_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
        learning_rate: float = 0.001,
        scheduler_factor: float = 0.5,
        scheduler_patience: int = 10,
        scaler=None,
    ):
        """
        Args:
            num_nodes: Number of nodes (sensors) in the graph
            in_steps: Number of input time steps (default: 12)
            out_steps: Number of output time steps (default: 12)
            steps_per_day: Number of time steps per day (default: 288 for 5-min intervals)
            input_dim: Input feature dimension (default: 3)
            output_dim: Output feature dimension (default: 1)
            input_embedding_dim: Dimension of input embedding (default: 24)
            tod_embedding_dim: Dimension of time-of-day embedding (default: 24)
            dow_embedding_dim: Dimension of day-of-week embedding (default: 24)
            nid_embedding_dim: Dimension of node ID embedding (default: 24)
            col_embedding_dim: Dimension of column embedding (default: 80)
            feed_forward_dim: Dimension of feed-forward layer (default: 256)
            num_heads: Number of attention heads (default: 4)
            num_layers: Number of transformer layers (default: 3)
            dropout: Dropout rate (default: 0.1)
            learning_rate: Learning rate (default: 0.001)
            scheduler_factor: LR scheduler factor (default: 0.5)
            scheduler_patience: LR scheduler patience (default: 10)
            scaler: MinMaxScaler instance for inverse transform (optional, default: None)
        """
        super().__init__()
        self.save_hyperparameters(ignore=["scaler"])
        
        # Store scaler for inverse transform (not saved in checkpoints)
        self.scaler = scaler

        # Initialize the MLCAFormer model
        self.model = MLCAFormer(
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
        """Configure optimizer with ReduceLROnPlateau scheduler"""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=self.scheduler_factor,
            patience=self.scheduler_patience,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        # Flatten to (batch * out_steps * num_nodes, 1)
        flat_data = data.reshape(-1, 1)
        # Inverse transform
        unscaled = self.scaler.inverse_transform(flat_data)
        # Restore original shape
        return unscaled.reshape(original_shape)

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Training step"""
        x, y = batch
        y_hat: torch.Tensor = self(x)

        if torch.isnan(y_hat).any():
            print("NaN detected in model output!")

        # MLCAFormer output: (batch, out_steps, num_nodes, output_dim)
        # Squeeze output_dim if it's 1
        if y_hat.dim() == 4 and y_hat.size(-1) == 1:
            y_hat = y_hat.squeeze(-1)  # (batch, out_steps, num_nodes)

        # Reshape y if needed to match y_hat shape
        if y.dim() == 4 and y.size(-1) == 1:
            y = y.squeeze(-1)

        loss: torch.Tensor = self.criterion(y_hat, y)

        self.log("train_loss", loss, on_step=True, on_epoch=True)

        return loss

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Validation step"""
        x, y = batch
        y_hat: torch.Tensor = self(x)

        # MLCAFormer output: (batch, out_steps, num_nodes, output_dim)
        if y_hat.dim() == 4 and y_hat.size(-1) == 1:
            y_hat = y_hat.squeeze(-1)

        if y.dim() == 4 and y.size(-1) == 1:
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
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Test step with missing mask support"""
        x, y, y_is_missing = batch
        y_hat: torch.Tensor = self(x)

        # MLCAFormer output: (batch, out_steps, num_nodes, output_dim)
        if y_hat.dim() == 4 and y_hat.size(-1) == 1:
            y_hat = y_hat.squeeze(-1)

        if y.dim() == 4 and y.size(-1) == 1:
            y = y.squeeze(-1)

        loss: torch.Tensor = self.criterion(y_hat, y)

        # Store outputs for epoch-end metrics
        self.test_outputs.append(
            {
                "y_true": y.cpu().numpy(),
                "y_pred": y_hat.cpu().numpy(),
                "y_is_missing": y_is_missing.cpu().numpy(),
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
        y_is_missing = np.concatenate([x["y_is_missing"] for x in self.test_outputs], axis=0)

        # Calculate metrics on scaled data (all data) - print only
        mae_scaled_all = mean_absolute_error(y_true.flatten(), y_pred.flatten())
        rmse_scaled_all = np.sqrt(mean_squared_error(y_true.flatten(), y_pred.flatten()))

        # MAPE with zero-division handling
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        mask_all = y_true_flat != 0
        mape_scaled_all = (
            np.mean(np.abs((y_true_flat[mask_all] - y_pred_flat[mask_all]) / y_true_flat[mask_all]))
            * 100
        ) if mask_all.any() else 0.0

        print(f"\nTest Results (Scaled - All Data):")
        print(f"MAE: {mae_scaled_all:.4f}")
        print(f"RMSE: {rmse_scaled_all:.4f}")
        print(f"MAPE: {mape_scaled_all:.4f}%")
        
        # Calculate metrics on non-missing (original) data only - log these
        y_is_missing_flat = y_is_missing.flatten()
        non_missing_mask = ~y_is_missing_flat
        
        if non_missing_mask.any():
            y_true_non_missing = y_true_flat[non_missing_mask]
            y_pred_non_missing = y_pred_flat[non_missing_mask]
            
            mae_scaled = mean_absolute_error(y_true_non_missing, y_pred_non_missing)
            rmse_scaled = np.sqrt(mean_squared_error(y_true_non_missing, y_pred_non_missing))
            
            non_zero_mask = y_true_non_missing != 0
            mape_scaled = (
                np.mean(np.abs((y_true_non_missing[non_zero_mask] - y_pred_non_missing[non_zero_mask]) / y_true_non_missing[non_zero_mask]))
                * 100
            ) if non_zero_mask.any() else 0.0
            
            # Log scaled metrics with _scaled suffix
            self.log("test_mae_scaled", mae_scaled)
            self.log("test_rmse_scaled", rmse_scaled)
            self.log("test_mape_scaled", float(mape_scaled))
            
            print(f"\nTest Results (Scaled - Non-Missing Data Only):")
            print(f"MAE: {mae_scaled:.4f}")
            print(f"RMSE: {rmse_scaled:.4f}")
            print(f"MAPE: {mape_scaled:.4f}%")
            print(f"Non-missing ratio: {non_missing_mask.sum() / len(y_is_missing_flat) * 100:.2f}%")
        
        # Calculate unscaled metrics if scaler is provided
        if self.scaler is not None:
            y_true_unscaled = self._inverse_transform(y_true)
            y_pred_unscaled = self._inverse_transform(y_pred)
            
            # All data metrics (unscaled) - print only
            mae_unscaled_all = mean_absolute_error(y_true_unscaled.flatten(), y_pred_unscaled.flatten())
            rmse_unscaled_all = np.sqrt(mean_squared_error(y_true_unscaled.flatten(), y_pred_unscaled.flatten()))
            
            # MAPE with zero-division handling
            y_true_unscaled_flat = y_true_unscaled.flatten()
            y_pred_unscaled_flat = y_pred_unscaled.flatten()
            mask_unscaled_all = y_true_unscaled_flat != 0
            mape_unscaled_all = (
                np.mean(np.abs((y_true_unscaled_flat[mask_unscaled_all] - y_pred_unscaled_flat[mask_unscaled_all]) / y_true_unscaled_flat[mask_unscaled_all]))
                * 100
            ) if mask_unscaled_all.any() else 0.0
            
            print(f"\nTest Results (Original Scale - All Data):")
            print(f"MAE: {mae_unscaled_all:.4f}")
            print(f"RMSE: {rmse_unscaled_all:.4f}")
            print(f"MAPE: {mape_unscaled_all:.4f}%")
            
            # Non-missing data metrics on original scale - log these as primary metrics
            if non_missing_mask.any():
                y_true_unscaled_non_missing = y_true_unscaled_flat[non_missing_mask]
                y_pred_unscaled_non_missing = y_pred_unscaled_flat[non_missing_mask]
                
                mae_unscaled = mean_absolute_error(y_true_unscaled_non_missing, y_pred_unscaled_non_missing)
                rmse_unscaled = np.sqrt(mean_squared_error(y_true_unscaled_non_missing, y_pred_unscaled_non_missing))
                
                non_zero_mask_unscaled = y_true_unscaled_non_missing != 0
                mape_unscaled = (
                    np.mean(np.abs((y_true_unscaled_non_missing[non_zero_mask_unscaled] - y_pred_unscaled_non_missing[non_zero_mask_unscaled]) / y_true_unscaled_non_missing[non_zero_mask_unscaled]))
                    * 100
                ) if non_zero_mask_unscaled.any() else 0.0
                
                # Log unscaled non-missing metrics as primary metrics
                self.log("test_mae", mae_unscaled)
                self.log("test_rmse", rmse_unscaled)
                self.log("test_mape", float(mape_unscaled))
                
                print(f"\nTest Results (Original Scale - Non-Missing Data Only):")
                print(f"MAE: {mae_unscaled:.4f}")
                print(f"RMSE: {rmse_unscaled:.4f}")
                print(f"MAPE: {mape_unscaled:.4f}%")

        # Clear outputs
        self.test_outputs.clear()
