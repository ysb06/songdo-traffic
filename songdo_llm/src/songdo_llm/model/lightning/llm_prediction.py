import logging
from typing import Dict, List
import os

from ollama import chat
import lightning as L
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import (
    DebertaV2ForSequenceClassification,
    DebertaV2Tokenizer,
    DebertaV2PreTrainedModel,
    DebertaV2Model,
)
import yaml

from ..base.rnn import TrafficLSTM
from ..base.llm import LLMWeight


class LLMTrafficPredictionModule(L.LightningModule):
    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 32,
        num_layers: int = 1,
        output_dim: int = 1,
        learning_rate: float = 1e-3,
        lr_step_size: int = 100,
        lr_gamma: float = 0.5,
        llm_name: str = "deepseek-r1:8b",
        classifier_name: str = "microsoft/deberta-v3-base",
        llm_cache_path: str = "./output/llm_cache",
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        # Traffic Model
        self.traffic_model = TrafficLSTM(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            output_dim=output_dim,
        )

        # LLM Model
        self.llm_model = LLMWeight(
            llm_name=llm_name,
            classifier_name=classifier_name,
            llm_cache_path=llm_cache_path,
        )

        self.lr = learning_rate
        self.lr_step_size = lr_step_size
        self.lr_gamma = lr_gamma

        self.criterion = nn.MSELoss()

    def forward(self, x_1: torch.Tensor, x_2: List[pd.Timestamp]) -> torch.Tensor:
        y_2_hat = self.llm_model(x_2)

        return self.traffic_model(x_1) * y_2_hat

    def training_step(self, batch, _):
        x_batch, y_batch, _, _ = batch
        outputs = self(x_batch, y_batch)
        loss = self.criterion(outputs, y_batch)

        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, _):
        x_batch, y_batch, _, _ = batch
        outputs = self(x_batch, y_batch)
        loss = self.criterion(outputs, y_batch)
        self.log("val_loss", loss, prog_bar=True)

        return loss

    def test_step(self, batch, _):
        x_batch, y_batch, _, _ = batch
        outputs = self(x_batch, y_batch)
        loss = self.criterion(outputs, y_batch)
        self.log("test_loss", loss, prog_bar=True)

        return loss

    def predict_step(self, batch, _):
        x_batch: torch.Tensor = batch[0]
        y_batch: torch.Tensor = batch[1]
        preds: torch.Tensor = self(x_batch, y_batch)

        return preds.cpu(), y_batch.cpu()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.traffic_model.parameters() + self.llm_model.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.lr_step_size,
            gamma=self.lr_gamma,
        )

        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
