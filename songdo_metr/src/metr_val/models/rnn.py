import torch
import torch.nn as nn
import lightning as L
from typing import Tuple
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


class LSTMBase(nn.Module):
    """
    Basic PyTorch LSTM model for traffic prediction.
    """

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 64,
        num_layers: int = 2,
        output_size: int = 1,
        dropout_rate: float = 0.2,
    ):
        super(LSTMBase, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
        )

        # Fully connected layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the LSTM and fully connected layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, input_size)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size)
        """
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]  # Get the last time step output
        output = self.fc(last_output)  # Fully connected layer

        return output


class LSTMTrainer(L.LightningModule):
    """
    PyTorch Lightning module for training LSTMBase model for traffic prediction
    """

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 64,
        num_layers: int = 2,
        output_size: int = 1,
        learning_rate: float = 0.001,
        dropout_rate: float = 0.2,
        scheduler_factor: float = 0.5,
        scheduler_patience: int = 10,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Initialize the LSTMBase model
        self.model = LSTMBase(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=output_size,
            dropout_rate=dropout_rate,
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
        """Configure optimizer"""
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
        """
        Forward pass

        Args:
            x: Input tensor of shape (batch_size, seq_length, input_size)

        Returns:
            Output tensor of shape (batch_size, output_size)
        """
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

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        
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

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

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

        # Calculate metrics
        mae = mean_absolute_error(y_true.flatten(), y_pred.flatten())
        rmse = np.sqrt(mean_squared_error(y_true.flatten(), y_pred.flatten()))

        self.log("val_mae", mae, prog_bar=True)
        self.log("val_rmse", rmse, prog_bar=True)

        # Clear outputs for next epoch
        self.validation_outputs.clear()

    def on_test_epoch_end(self):
        """Calculate test metrics at the end of epoch"""
        if len(self.test_outputs) == 0:
            return

        # Concatenate all predictions and targets
        y_true = np.concatenate([x["y_true"] for x in self.test_outputs], axis=0)
        y_pred = np.concatenate([x["y_pred"] for x in self.test_outputs], axis=0)

        # Calculate metrics
        mae = mean_absolute_error(y_true.flatten(), y_pred.flatten())
        rmse = np.sqrt(mean_squared_error(y_true.flatten(), y_pred.flatten()))
        mape = (
            np.mean(np.abs((y_true.flatten() - y_pred.flatten()) / y_true.flatten()))
            * 100
        )

        self.log("test_mae", mae)
        self.log("test_rmse", rmse)
        self.log("test_mape", mape)

        print(f"\nTest Results:")
        print(f"MAE: {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAPE: {mape:.4f}%")

        # Clear outputs
        self.test_outputs.clear()
