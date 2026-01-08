"""
TASSGN Lightning Modules for 5-Phase Training Pipeline.

Phase 1: STIDEncoderLightningModule - Self-supervised encoder pretraining
Phase 2: (Data generation - not a Lightning module)
Phase 3: PredictorLightningModule - Label predictor training  
Phase 4: (Data generation - not a Lightning module)
Phase 5: TASSGNLightningModule - Final traffic prediction
"""

from typing import Any, Dict, List, Optional, Tuple

import lightning as L
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

from .encoder import STIDEncoder
from .predictor import Predictor
from .TASSGN import TASSGN


class STIDEncoderLightningModule(L.LightningModule):
    """Phase 1: Self-supervised encoder pretraining.
    
    The encoder is trained to reconstruct randomly masked input sequences.
    This learns spatial-temporal representations that will be used by the Predictor.
    
    Input: x = (B, T, N, 3) [traffic, time_of_day, day_of_week]
    Output: reconstructed = (B, T, N, 1) [traffic channel only]
    Target: y = (B, T, N, 3) - original unmasked data
    Loss: MSELoss on traffic channel
    """
    
    def __init__(
        self,
        num_nodes: int,
        input_len: int = 12,
        output_len: int = 12,
        input_dim: int = 1,
        hid_dim: int = 32,
        num_layers: int = 3,
        time_of_day_size: int = 288,
        day_of_week_size: int = 7,
        mask_ratio: float = 0.5,
        learning_rate: float = 0.002,
    ):
        """Initialize STIDEncoder Lightning Module.
        
        Args:
            num_nodes: Number of nodes (sensors) in the graph.
            input_len: Length of input sequence.
            output_len: Length of output sequence (same as input_len for reconstruction).
            input_dim: Input dimension (traffic flow only, excluding temporal features).
            hid_dim: Hidden dimension.
            num_layers: Number of MLP encoding layers.
            time_of_day_size: Number of time steps per day.
            day_of_week_size: Number of days per week.
            mask_ratio: Ratio of time steps to mask for self-supervised training.
            learning_rate: Learning rate for optimizer.
        """
        super().__init__()
        self.save_hyperparameters()
        
        self.learning_rate = learning_rate
        
        self.model = STIDEncoder(
            num_nodes=num_nodes,
            input_len=input_len,
            output_len=output_len,
            input_dim=input_dim,
            hid_dim=hid_dim,
            num_layers=num_layers,
            time_of_day_size=time_of_day_size,
            day_of_week_size=day_of_week_size,
            mask_ratio=mask_ratio,
        )
        
        self.criterion = nn.MSELoss()
        
        # For validation metrics
        self.validation_outputs: List[Dict[str, Any]] = []
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for reconstruction.
        
        Args:
            x: Input tensor of shape (B, T, N, 3).
            
        Returns:
            Reconstructed tensor of shape (B, T, N, 1).
        """
        return self.model.pretrain(x)
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }
    
    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Training step with masked reconstruction.
        
        Args:
            batch: Tuple of (x, y) where both are (B, T, N, 3).
                   x will be masked internally, y is the reconstruction target.
            batch_idx: Batch index.
            
        Returns:
            Loss value.
        """
        x, y = batch
        
        # Forward pass (masking happens inside pretrain)
        y_hat = self(x)  # Model outputs (B, output_len, N, 1)
        
        # Target is the traffic channel of original data
        y_target = y[..., 0:1]  # (B, T, N, 1)
        
        # Model regression_layer outputs (B, output_len, N, 1) which matches target shape
        # No transpose needed
        
        loss = self.criterion(y_hat, y_target)
        
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Validation step.
        
        Args:
            batch: Tuple of (x, y).
            batch_idx: Batch index.
            
        Returns:
            Loss value.
        """
        x, y = batch
        
        # For validation, use encode without masking to evaluate representation quality
        # But we still use pretrain to maintain consistency
        y_hat = self(x)  # Model outputs (B, output_len, N, 1)
        
        y_target = y[..., 0:1]  # (B, T, N, 1)
        # No transpose needed - shapes already match
        
        loss = self.criterion(y_hat, y_target)
        
        self.validation_outputs.append({
            "loss": loss.item(),
        })
        
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def on_validation_epoch_end(self):
        """Calculate validation metrics at the end of epoch."""
        if len(self.validation_outputs) == 0:
            return
        
        avg_loss = float(np.mean([x["loss"] for x in self.validation_outputs]))
        self.log("val_loss_avg", avg_loss)
        
        self.validation_outputs.clear()
    
    def get_encoder_state_dict(self) -> Dict[str, torch.Tensor]:
        """Get the encoder state dict for loading into Predictor.
        
        Returns:
            State dict of the encoder model.
        """
        return self.model.state_dict()


class PredictorLightningModule(L.LightningModule):
    """Phase 3: Label predictor training.
    
    The predictor uses the pretrained encoder to predict cluster labels
    for future time series based on history data.
    
    Input: x = (B, T, N, 3) [history window]
    Output: logits = (B, N, C) [C = num_clusters]
    Target: labels = (B, N) [cluster labels, LongTensor]
    Loss: CrossEntropyLoss
    """
    
    def __init__(
        self,
        num_nodes: int,
        input_len: int = 12,
        input_dim: int = 1,
        hid_dim: int = 32,
        num_clusters: int = 20,
        num_layers: int = 3,
        time_of_day_size: int = 288,
        day_of_week_size: int = 7,
        encoder_state_dict: Optional[Dict[str, torch.Tensor]] = None,
        learning_rate: float = 0.002,
    ):
        """Initialize Predictor Lightning Module.
        
        Args:
            num_nodes: Number of nodes (sensors) in the graph.
            input_len: Length of input sequence.
            input_dim: Input dimension (traffic flow only).
            hid_dim: Hidden dimension.
            num_clusters: Number of cluster labels (output dimension).
            num_layers: Number of MLP encoding layers.
            time_of_day_size: Number of time steps per day.
            day_of_week_size: Number of days per week.
            encoder_state_dict: Pretrained encoder weights from Phase 1.
            learning_rate: Learning rate for optimizer.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["encoder_state_dict"])
        
        self.learning_rate = learning_rate
        self.num_clusters = num_clusters
        
        self.model = Predictor(
            num_nodes=num_nodes,
            input_len=input_len,
            input_dim=input_dim,
            hid_dim=hid_dim,
            out_dim=num_clusters,
            num_layers=num_layers,
            time_of_day_size=time_of_day_size,
            day_of_week_size=day_of_week_size,
        )
        
        # Load pretrained encoder weights if provided
        if encoder_state_dict is not None:
            self.load_encoder_weights(encoder_state_dict)
        
        self.criterion = nn.CrossEntropyLoss()
        
        # For metrics
        self.validation_outputs: List[Dict[str, Any]] = []
        self.test_outputs: List[Dict[str, Any]] = []
    
    def load_encoder_weights(self, encoder_state_dict: Dict[str, torch.Tensor]):
        """Load pretrained encoder weights into the Predictor's encoder.
        
        Note: Only loads weights that are compatible between Phase 1 STIDEncoder
        and Predictor's internal encoder. The regression_layer is excluded because
        Phase 1 encoder has output_len=12 while Predictor's encoder has output_len=num_clusters.
        
        Args:
            encoder_state_dict: State dict from STIDEncoder.
        """
        # Filter out regression_layer weights (size mismatch between Phase 1 and Phase 3)
        # Phase 1: regression_layer outputs [12] (out_steps)
        # Phase 3: regression_layer outputs [num_clusters] (different purpose)
        filtered_state_dict = {
            k: v for k, v in encoder_state_dict.items()
            if not k.startswith("regression_layer")
        }
        
        # Load with strict=False to allow missing regression_layer
        self.model.encoder.load_state_dict(filtered_state_dict, strict=False)
        print(f"Loaded pretrained encoder weights into Predictor (excluded regression_layer).")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for label prediction.
        
        Args:
            x: Input tensor of shape (B, T, N, 3).
            
        Returns:
            Logits tensor of shape (B, N, C).
        """
        return self.model(x)
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }
    
    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Training step for label prediction.
        
        Args:
            batch: Tuple of (x, labels) where x is (B, T, N, 3) and labels is (B, N).
            batch_idx: Batch index.
            
        Returns:
            Loss value.
        """
        x, labels = batch
        
        # Forward pass
        logits = self(x)  # (B, N, C)
        
        # Reshape for CrossEntropyLoss: (B*N, C) and (B*N,)
        batch_size, num_nodes, num_classes = logits.shape
        logits_flat = logits.view(-1, num_classes)
        labels_flat = labels.view(-1)
        
        loss = self.criterion(logits_flat, labels_flat)
        
        # Calculate accuracy
        preds = torch.argmax(logits_flat, dim=-1)
        accuracy = (preds == labels_flat).float().mean()
        
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", accuracy, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Validation step.
        
        Args:
            batch: Tuple of (x, labels).
            batch_idx: Batch index.
            
        Returns:
            Loss value.
        """
        x, labels = batch
        
        logits = self(x)
        
        batch_size, num_nodes, num_classes = logits.shape
        logits_flat = logits.view(-1, num_classes)
        labels_flat = labels.view(-1)
        
        loss = self.criterion(logits_flat, labels_flat)
        
        preds = torch.argmax(logits_flat, dim=-1)
        accuracy = (preds == labels_flat).float().mean()
        
        self.validation_outputs.append({
            "loss": loss.item(),
            "accuracy": accuracy.item(),
            "correct": (preds == labels_flat).sum().item(),
            "total": labels_flat.numel(),
        })
        
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", accuracy, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def on_validation_epoch_end(self):
        """Calculate validation metrics at the end of epoch."""
        if len(self.validation_outputs) == 0:
            return
        
        total_correct = sum(x["correct"] for x in self.validation_outputs)
        total_samples = sum(x["total"] for x in self.validation_outputs)
        epoch_accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        
        self.log("val_acc_epoch", epoch_accuracy)
        
        self.validation_outputs.clear()
    
    def get_predictor_model(self) -> Predictor:
        """Get the predictor model for generating sampling indices.
        
        Returns:
            The Predictor model.
        """
        return self.model


class TASSGNLightningModule(L.LightningModule):
    """Phase 5: Final TASSGN traffic prediction.
    
    The full TASSGN model uses history data and self-sampled similar patterns
    to predict future traffic.
    
    Input: 
        - history_data = (B, T, N, 3)
        - sample_data = (B, S, T, N, 3) [S = num_samples]
    Output: prediction = (B, T_out, N, 1)
    Target: y = (B, T_out, N, 3) - use traffic channel only
    Loss: L1Loss (MAE)
    """
    
    def __init__(
        self,
        num_nodes: int,
        input_len: int = 12,
        output_len: int = 12,
        input_dim: int = 1,
        hid_dim: int = 32,
        num_samples: int = 7,
        num_layers: int = 3,
        num_blocks: int = 2,
        num_attention_heads: int = 2,
        topk: int = 10,
        dropout: float = 0.1,
        time_of_day_size: int = 288,
        day_of_week_size: int = 7,
        learning_rate: float = 0.002,
        scaler: Optional[StandardScaler] = None,
    ):
        """Initialize TASSGN Lightning Module.
        
        Args:
            num_nodes: Number of nodes (sensors) in the graph.
            input_len: Length of input sequence.
            output_len: Length of output sequence.
            input_dim: Input dimension (traffic flow only).
            hid_dim: Hidden dimension.
            num_samples: Number of self-sampled similar patterns.
            num_layers: Number of MLP encoding layers.
            num_blocks: Number of TAGEncoder blocks.
            num_attention_heads: Number of attention heads in TAGEncoder.
            topk: Top-k for attention mechanism.
            dropout: Dropout rate.
            time_of_day_size: Number of time steps per day.
            day_of_week_size: Number of days per week.
            learning_rate: Learning rate for optimizer.
            scaler: StandardScaler for inverse transform (for unscaled metrics).
        """
        super().__init__()
        self.save_hyperparameters(ignore=["scaler"])
        
        self.learning_rate = learning_rate
        self.scaler = scaler
        
        self.model = TASSGN(
            num_nodes=num_nodes,
            input_len=input_len,
            output_len=output_len,
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
        )
        
        self.criterion = nn.L1Loss()  # MAE Loss
        
        # For metrics
        self.validation_outputs: List[Dict[str, Any]] = []
        self.test_outputs: List[Dict[str, Any]] = []
    
    def forward(
        self, history_data: torch.Tensor, sample_data: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass for traffic prediction.
        
        Args:
            history_data: Input tensor of shape (B, T, N, 3).
            sample_data: Sampled similar patterns of shape (B, S, T, N, 3).
            
        Returns:
            Prediction tensor of shape (B, T_out, N, 1).
        """
        return self.model(history_data, sample_data)
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }
    
    def _inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Inverse transform scaled data to original scale.
        
        Args:
            data: Scaled data array.
            
        Returns:
            Unscaled data array.
        """
        if self.scaler is None:
            return data
        
        original_shape = data.shape
        data_flat = data.reshape(-1, 1)
        data_unscaled = self.scaler.inverse_transform(data_flat)
        return data_unscaled.reshape(original_shape)
    
    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Training step for traffic prediction.
        
        Args:
            batch: Tuple of (x, y, sample_data) where:
                   - x is (B, T, N, 3)
                   - y is (B, T_out, N, 3)
                   - sample_data is (B, S, T, N, 3)
            batch_idx: Batch index.
            
        Returns:
            Loss value.
        """
        x, y, sample_data = batch
        
        # Forward pass
        y_hat = self(x, sample_data)  # (B, T_out, N, 1)
        
        # Target is the traffic channel
        y_target = y[..., 0:1]  # (B, T_out, N, 1)
        
        # Reshape prediction to match target
        # Model output: (B, T_out, N, 1) after regression_layer with proper transpose
        y_hat = y_hat.squeeze(-1).transpose(1, 2).unsqueeze(-1)  # (B, T_out, N, 1)
        
        loss = self.criterion(y_hat, y_target)
        
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Validation step.
        
        Args:
            batch: Tuple of (x, y, sample_data).
            batch_idx: Batch index.
            
        Returns:
            Loss value.
        """
        x, y, sample_data = batch
        
        y_hat = self(x, sample_data)
        y_target = y[..., 0:1]
        y_hat = y_hat.squeeze(-1).transpose(1, 2).unsqueeze(-1)
        
        loss = self.criterion(y_hat, y_target)
        
        self.validation_outputs.append({
            "y_true": y_target.cpu().numpy(),
            "y_pred": y_hat.cpu().numpy(),
            "loss": loss.item(),
        })
        
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Test step.
        
        Args:
            batch: Tuple of (x, y, sample_data).
            batch_idx: Batch index.
            
        Returns:
            Loss value.
        """
        x, y, sample_data = batch
        
        y_hat = self(x, sample_data)
        y_target = y[..., 0:1]
        y_hat = y_hat.squeeze(-1).transpose(1, 2).unsqueeze(-1)
        
        loss = self.criterion(y_hat, y_target)
        
        self.test_outputs.append({
            "y_true": y_target.cpu().numpy(),
            "y_pred": y_hat.cpu().numpy(),
            "loss": loss.item(),
        })
        
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def on_validation_epoch_end(self):
        """Calculate validation metrics at the end of epoch."""
        if len(self.validation_outputs) == 0:
            return
        
        y_true = np.concatenate([x["y_true"] for x in self.validation_outputs], axis=0)
        y_pred = np.concatenate([x["y_pred"] for x in self.validation_outputs], axis=0)
        
        # Metrics on scaled data
        mae_scaled = mean_absolute_error(y_true.flatten(), y_pred.flatten())
        rmse_scaled = np.sqrt(mean_squared_error(y_true.flatten(), y_pred.flatten()))
        
        self.log("val_mae", mae_scaled)
        self.log("val_rmse", rmse_scaled)
        
        self.validation_outputs.clear()
    
    def on_test_epoch_end(self):
        """Calculate test metrics at the end of epoch with inverse transform."""
        if len(self.test_outputs) == 0:
            return
        
        y_true = np.concatenate([x["y_true"] for x in self.test_outputs], axis=0)
        y_pred = np.concatenate([x["y_pred"] for x in self.test_outputs], axis=0)
        
        # Metrics on scaled data (for reference)
        mae_scaled = mean_absolute_error(y_true.flatten(), y_pred.flatten())
        rmse_scaled = np.sqrt(mean_squared_error(y_true.flatten(), y_pred.flatten()))
        
        self.log("test_mae_scaled", mae_scaled)
        self.log("test_rmse_scaled", rmse_scaled)
        
        print(f"\nTest Results (Scaled):")
        print(f"MAE: {mae_scaled:.4f}")
        print(f"RMSE: {rmse_scaled:.4f}")
        
        # Primary metrics: Inverse transform to original scale
        if self.scaler is not None:
            y_true_unscaled = self._inverse_transform(y_true)
            y_pred_unscaled = self._inverse_transform(y_pred)
            
            mae = mean_absolute_error(y_true_unscaled.flatten(), y_pred_unscaled.flatten())
            rmse = np.sqrt(mean_squared_error(y_true_unscaled.flatten(), y_pred_unscaled.flatten()))
            
            # MAPE with zero-division handling
            y_true_flat = y_true_unscaled.flatten()
            y_pred_flat = y_pred_unscaled.flatten()
            mask = y_true_flat != 0
            mape = (
                np.mean(np.abs((y_true_flat[mask] - y_pred_flat[mask]) / y_true_flat[mask]))
                * 100
            ) if mask.any() else 0.0
            
            self.log("test_mae", float(mae))
            self.log("test_rmse", float(rmse))
            self.log("test_mape", float(mape))
            
            print(f"\nTest Results (Original Scale):")
            print(f"MAE: {mae:.4f}")
            print(f"RMSE: {rmse:.4f}")
            print(f"MAPE: {mape:.4f}%")
        else:
            # If no scaler, use scaled metrics as primary
            self.log("test_mae", mae_scaled)
            self.log("test_rmse", rmse_scaled)
            
            # MAPE on scaled data
            y_true_flat = y_true.flatten()
            y_pred_flat = y_pred.flatten()
            mask = y_true_flat != 0
            mape = (
                np.mean(np.abs((y_true_flat[mask] - y_pred_flat[mask]) / y_true_flat[mask]))
                * 100
            ) if mask.any() else 0.0
            
            self.log("test_mape", float(mape))
            print(f"MAPE: {mape:.4f}%")
            print("\nWarning: No scaler provided. Metrics are on scaled data.")
        
        self.test_outputs.clear()
