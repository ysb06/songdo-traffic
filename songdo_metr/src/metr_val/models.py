"""
Basic RNN model for traffic prediction validation
"""

import torch
import torch.nn as nn
import lightning as L
from typing import Tuple
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


class BasicRNN(L.LightningModule):
    """
    Basic RNN model for traffic prediction using PyTorch Lightning
    """
    
    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 64,
        num_layers: int = 2,
        output_size: int = 1,
        learning_rate: float = 0.001,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # RNN layers
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Output layer
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
        
        # Metrics storage
        self.validation_outputs = []
        self.test_outputs = []
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, seq_length, input_size)
            
        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        # x shape: (batch_size, seq_length, input_size)
        lstm_out, _ = self.rnn(x)
        
        # Take the last output of the sequence
        # lstm_out shape: (batch_size, seq_length, hidden_size)
        last_output = lstm_out[:, -1, :]  # (batch_size, hidden_size)
        
        # Apply dropout
        last_output = self.dropout(last_output)
        
        # Final prediction
        output = self.fc(last_output)  # (batch_size, output_size)
        
        return output
    
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step"""
        x, y = batch
        y_hat = self(x)
        
        # Reshape y if needed
        if y.dim() > 2:
            y = y.squeeze(-1)  # Remove last dimension if it's 1
        
        loss = nn.MSELoss()(y_hat, y)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step"""
        x, y = batch
        y_hat = self(x)
        
        # Reshape y if needed
        if y.dim() > 2:
            y = y.squeeze(-1)
        
        loss = nn.MSELoss()(y_hat, y)
        
        # Store outputs for epoch-end metrics
        self.validation_outputs.append({
            'y_true': y.cpu().numpy(),
            'y_pred': y_hat.cpu().numpy(),
            'loss': loss.item()
        })
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Test step"""
        x, y = batch
        y_hat = self(x)
        
        # Reshape y if needed
        if y.dim() > 2:
            y = y.squeeze(-1)
        
        loss = nn.MSELoss()(y_hat, y)
        
        # Store outputs for epoch-end metrics
        self.test_outputs.append({
            'y_true': y.cpu().numpy(),
            'y_pred': y_hat.cpu().numpy(),
            'loss': loss.item()
        })
        
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        return loss
    
    def on_validation_epoch_end(self):
        """Calculate validation metrics at the end of epoch"""
        if len(self.validation_outputs) == 0:
            return
        
        # Concatenate all predictions and targets
        y_true = np.concatenate([x['y_true'] for x in self.validation_outputs], axis=0)
        y_pred = np.concatenate([x['y_pred'] for x in self.validation_outputs], axis=0)
        
        # Calculate metrics
        mae = mean_absolute_error(y_true.flatten(), y_pred.flatten())
        rmse = np.sqrt(mean_squared_error(y_true.flatten(), y_pred.flatten()))
        
        self.log('val_mae', mae, prog_bar=True)
        self.log('val_rmse', rmse, prog_bar=True)
        
        # Clear outputs for next epoch
        self.validation_outputs.clear()
    
    def on_test_epoch_end(self):
        """Calculate test metrics at the end of epoch"""
        if len(self.test_outputs) == 0:
            return
        
        # Concatenate all predictions and targets
        y_true = np.concatenate([x['y_true'] for x in self.test_outputs], axis=0)
        y_pred = np.concatenate([x['y_pred'] for x in self.test_outputs], axis=0)
        
        # Calculate metrics
        mae = mean_absolute_error(y_true.flatten(), y_pred.flatten())
        rmse = np.sqrt(mean_squared_error(y_true.flatten(), y_pred.flatten()))
        mape = np.mean(np.abs((y_true.flatten() - y_pred.flatten()) / y_true.flatten())) * 100
        
        self.log('test_mae', mae)
        self.log('test_rmse', rmse)
        self.log('test_mape', mape)
        
        print(f"\nTest Results:")
        print(f"MAE: {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAPE: {mape:.4f}%")
        
        # Clear outputs
        self.test_outputs.clear()
    
    def configure_optimizers(self):
        """Configure optimizer"""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
            }
        }