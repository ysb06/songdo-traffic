from typing import List, Optional

import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class SongdoTrafficLightning(L.LightningModule):
    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 32,
        num_layers: int = 1,
        output_dim: int = 1,
        learning_rate: float = 1e-3,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        # 기존 RNN 모델 사용
        self.model = TrafficRNN(
            input_dim=self.hparams.input_dim,
            hidden_dim=self.hparams.hidden_dim,
            num_layers=self.hparams.num_layers,
            output_dim=self.hparams.output_dim,
        )
        self.criterion = nn.MSELoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, _):
        x_batch, y_batch = batch
        outputs = self.model(x_batch)
        loss = self.criterion(outputs, y_batch)

        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, _):
        x_batch, y_batch = batch
        outputs = self(x_batch)
        loss = self.criterion(outputs, y_batch)
        self.log("val_loss", loss, prog_bar=True)

        return loss

    def test_step(self, batch, _):
        x_batch, y_batch = batch
        outputs = self(x_batch)
        loss = self.criterion(outputs, y_batch)
        self.log("test_loss", loss, prog_bar=True)

        return loss
    
    def predict_step(self, batch, _):
        x_batch: torch.Tensor = batch[0]
        y_batch: torch.Tensor = batch[1]
        preds: torch.Tensor = self(x_batch)
        
        return preds.cpu(), y_batch.cpu()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.hparams.learning_rate)

        return optimizer


class TrafficRNN(nn.Module):
    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 32,
        num_layers: int = 1,
        output_dim: int = 1,
    ):
        super(TrafficRNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])

        return out


class TrafficLSTM(nn.Module):
    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 32,
        num_layers: int = 1,
        output_dim: int = 1,
    ):
        super(TrafficLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])

        return out
