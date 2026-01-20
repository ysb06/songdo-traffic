"""PyTorch Lightning module for DCRNN model training."""

import logging
from typing import Optional, Tuple

import lightning as L
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .dcrnn_model import DCRNNModel


class DCRNNLightningModule(L.LightningModule):
    """PyTorch Lightning module for training DCRNN model for traffic prediction.
    
    This module wraps the DCRNNModel for use with PyTorch Lightning,
    handling training, validation, and testing with proper metrics logging.
    
    Args:
        adj_mx: Adjacency matrix (numpy array)
        num_nodes: Number of nodes in the graph
        input_dim: Input feature dimension (default: 2 for traffic + time_in_day)
        output_dim: Output feature dimension (default: 1 for traffic only)
        seq_len: Input sequence length (default: 12)
        horizon: Prediction horizon (default: 12)
        rnn_units: Number of RNN hidden units (default: 64)
        num_rnn_layers: Number of RNN layers (default: 2)
        max_diffusion_step: Max diffusion step for graph convolution (default: 2)
        filter_type: Graph filter type ('laplacian', 'random_walk', 'dual_random_walk')
        use_curriculum_learning: Whether to use curriculum learning (default: True)
        cl_decay_steps: Curriculum learning decay steps (default: 2000)
        learning_rate: Learning rate (default: 0.01)
        weight_decay: Weight decay for optimizer (default: 0)
        scheduler_step_size: LR scheduler step size (default: 10)
        scheduler_gamma: LR scheduler gamma (default: 0.95)
        scaler: StandardScaler for inverse transform (optional)
    """

    def __init__(
        self,
        adj_mx: np.ndarray,
        num_nodes: int,
        input_dim: int = 2,
        output_dim: int = 1,
        seq_len: int = 12,
        horizon: int = 12,
        rnn_units: int = 64,
        num_rnn_layers: int = 2,
        max_diffusion_step: int = 2,
        filter_type: str = "dual_random_walk",
        use_curriculum_learning: bool = True,
        cl_decay_steps: int = 2000,
        learning_rate: float = 0.01,
        weight_decay: float = 0,
        scheduler_step_size: int = 10,
        scheduler_gamma: float = 0.95,
        scaler=None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["adj_mx", "scaler"])
        
        # Store scaler for inverse transform
        self.scaler = scaler
        
        # Create logger for DCRNN model
        self._logger = logging.getLogger(__name__)
        
        # Model kwargs for DCRNNModel
        model_kwargs = {
            "num_nodes": num_nodes,
            "input_dim": input_dim,
            "output_dim": output_dim,
            "seq_len": seq_len,
            "horizon": horizon,
            "rnn_units": rnn_units,
            "num_rnn_layers": num_rnn_layers,
            "max_diffusion_step": max_diffusion_step,
            "filter_type": filter_type,
            "use_curriculum_learning": use_curriculum_learning,
            "cl_decay_steps": cl_decay_steps,
        }
        
        # Initialize DCRNN model
        self.model = DCRNNModel(adj_mx, self._logger, **model_kwargs)
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Learning rate and scheduler parameters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.scheduler_step_size = scheduler_step_size
        self.scheduler_gamma = scheduler_gamma
        
        # Track batches seen for curriculum learning
        self.batches_seen = 0
        
        # Flag to track if optimizer needs to be reconfigured after first forward pass
        # DCGRUCell dynamically registers parameters during first forward pass
        self._optimizer_reconfigured = False
        
        # Metrics storage
        self.validation_outputs = []
        self.test_outputs = []

    def configure_optimizers(self):
        """Configure optimizer with MultiStepLR scheduler.
        
        Note: DCGRUCell dynamically registers parameters during the first forward pass.
        The optimizer will be reconfigured after the first training step to include
        all dynamically registered parameters.
        
        Uses MultiStepLR with milestones [20, 30, 40, 50] and gamma 0.1 as in original DCRNN.
        """
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            eps=1e-8,  # DCRNN original uses epsilon=1e-8
        )
        # Original DCRNN uses MultiStepLR with milestones
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[20, 30, 40, 50],  # Standard DCRNN milestones
            gamma=0.1,  # Reduce LR by 10x at each milestone
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }

    def forward(
        self,
        inputs: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        batches_seen: Optional[int] = None,
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            inputs: shape (seq_len, batch_size, num_nodes * input_dim)
            labels: shape (horizon, batch_size, num_nodes * output_dim)
            batches_seen: Number of batches seen (for curriculum learning)
            
        Returns:
            outputs: shape (horizon, batch_size, num_nodes * output_dim)
        """
        return self.model(inputs, labels, batches_seen)

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Training step with curriculum learning support.
        
        Note: DCGRUCell dynamically registers parameters during the first forward pass.
        After the first batch, we need to reinitialize the optimizer to include these
        new parameters. This is done by calling self.trainer.strategy.setup_optimizers().
        """
        x, y = batch
        # x: (seq_len, batch_size, num_nodes * input_dim)
        # y: (horizon, batch_size, num_nodes * output_dim)
        
        # Forward pass with curriculum learning
        y_hat = self(x, y, self.batches_seen)
        
        # CRITICAL: After the first forward pass, DCGRUCell has dynamically registered
        # its parameters. We need to reconfigure the optimizer to include them.
        if self.batches_seen == 0 and not self._optimizer_reconfigured:
            self._optimizer_reconfigured = True
            # Reinitialize optimizers to include dynamically registered parameters
            self.trainer.strategy.setup_optimizers(self.trainer)
            self._logger.info(
                f"Optimizer reconfigured with {sum(p.numel() for p in self.parameters() if p.requires_grad)} parameters"
            )
        
        self.batches_seen += 1
        
        # Calculate loss
        loss = self.criterion(y_hat, y)
        
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Validation step."""
        x, y = batch
        
        # Forward pass (no curriculum learning during validation)
        y_hat = self(x)
        
        # Calculate loss
        loss = self.criterion(y_hat, y)
        
        # Store outputs for epoch-end metrics
        # Reshape from (horizon, batch, num_nodes * output_dim) to (batch, horizon, num_nodes)
        y_np = y.permute(1, 0, 2).cpu().numpy()
        y_hat_np = y_hat.permute(1, 0, 2).cpu().numpy()
        
        self.validation_outputs.append({
            "y_true": y_np,
            "y_pred": y_hat_np,
            "loss": loss.item(),
        })
        
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss

    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Test step."""
        x, y = batch
        
        # Forward pass
        y_hat = self(x)
        
        # Calculate loss
        loss = self.criterion(y_hat, y)
        
        # Store outputs for epoch-end metrics
        y_np = y.permute(1, 0, 2).cpu().numpy()
        y_hat_np = y_hat.permute(1, 0, 2).cpu().numpy()
        
        self.test_outputs.append({
            "y_true": y_np,
            "y_pred": y_hat_np,
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
        
        self.log("val_mae", mae_scaled)
        self.log("val_rmse", rmse_scaled)
        
        # Calculate metrics for different horizons (3, 6, 12 steps)
        for horizon_idx, horizon_name in [(2, "3"), (5, "6"), (11, "12")]:
            if y_true.shape[1] > horizon_idx:
                y_true_h = y_true[:, horizon_idx, :]
                y_pred_h = y_pred[:, horizon_idx, :]
                mae_h = mean_absolute_error(y_true_h.flatten(), y_pred_h.flatten())
                self.log(f"val_mae_horizon{horizon_name}", mae_h)
        
        # Clear outputs for next epoch
        self.validation_outputs.clear()

    def on_test_epoch_end(self):
        """Calculate test metrics at the end of epoch.
        
        Metrics are computed on the original (unscaled) data using inverse transform.
        """
        if len(self.test_outputs) == 0:
            return
        
        # Concatenate all predictions and targets
        y_true = np.concatenate([x["y_true"] for x in self.test_outputs], axis=0)
        y_pred = np.concatenate([x["y_pred"] for x in self.test_outputs], axis=0)
        
        # Inverse transform to get original scale values
        if self.scaler is not None:
            # Store original shapes
            original_shape = y_true.shape
            
            # Flatten, inverse transform, and reshape back
            y_true_flat = y_true.reshape(-1, 1)
            y_pred_flat = y_pred.reshape(-1, 1)
            
            y_true_unscaled = self.scaler.inverse_transform(y_true_flat).reshape(original_shape)
            y_pred_unscaled = self.scaler.inverse_transform(y_pred_flat).reshape(original_shape)
        else:
            # No scaler provided, use original values
            y_true_unscaled = y_true
            y_pred_unscaled = y_pred
        
        # Calculate metrics on unscaled (original) data
        mae = mean_absolute_error(y_true_unscaled.flatten(), y_pred_unscaled.flatten())
        rmse = np.sqrt(mean_squared_error(y_true_unscaled.flatten(), y_pred_unscaled.flatten()))
        
        # MAPE with zero-division handling
        y_true_flat = y_true_unscaled.flatten()
        y_pred_flat = y_pred_unscaled.flatten()
        mask = y_true_flat != 0
        mape = (
            np.mean(np.abs((y_true_flat[mask] - y_pred_flat[mask]) / y_true_flat[mask])) * 100
            if mask.any()
            else 0.0
        )
        
        self.log("test_mae", mae)
        self.log("test_rmse", rmse)
        self.log("test_mape", float(mape))
        
        # Calculate metrics for different horizons
        print("\nTest Results (Original Scale):")
        print(f"Overall - MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.2f}%")
        
        for horizon_idx, horizon_name in [(2, "3"), (5, "6"), (11, "12")]:
            if y_true_unscaled.shape[1] > horizon_idx:
                y_true_h = y_true_unscaled[:, horizon_idx, :]
                y_pred_h = y_pred_unscaled[:, horizon_idx, :]
                mae_h = mean_absolute_error(y_true_h.flatten(), y_pred_h.flatten())
                rmse_h = np.sqrt(mean_squared_error(y_true_h.flatten(), y_pred_h.flatten()))
                self.log(f"test_mae_horizon{horizon_name}", mae_h)
                self.log(f"test_rmse_horizon{horizon_name}", rmse_h)
                print(f"Horizon {horizon_name} - MAE: {mae_h:.4f}, RMSE: {rmse_h:.4f}")
        
        # Clear outputs
        self.test_outputs.clear()
