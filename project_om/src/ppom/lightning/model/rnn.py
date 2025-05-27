import lightning as L
import torch
import torch.nn as nn
import torch.optim as optim


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


class TrafficVolumePredictionModule(L.LightningModule):
    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 32,
        num_layers: int = 1,
        output_dim: int = 1,
        learning_rate: float = 1e-3,
        lr_step_size: int = 100,
        lr_gamma: float = 0.5,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.model = TrafficLSTM(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            output_dim=output_dim,
        )
        self.criterion = nn.MSELoss()

        self.learning_rate = learning_rate
        self.lr_step_size = lr_step_size
        self.lr_gamma = lr_gamma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, _):
        if len(batch) == 2:
            x_batch, y_batch = batch
        else:
            x_batch, y_batch, _, _ = batch

        outputs = self.model(x_batch)
        loss = self.criterion(outputs, y_batch)

        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, _):
        if len(batch) == 2:
            x_batch, y_batch = batch
        else:
            x_batch, y_batch, _, _ = batch

        outputs = self(x_batch)
        loss = self.criterion(outputs, y_batch)
        self.log("val_loss", loss, prog_bar=True)

        return loss

    def test_step(self, batch, _):
        if len(batch) == 2:
            x_batch, y_batch = batch
        else:
            x_batch, y_batch, _, _ = batch

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
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.hparams.lr_step_size,
            gamma=self.hparams.lr_gamma,
        )

        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
