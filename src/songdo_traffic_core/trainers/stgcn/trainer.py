import os
import random

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from pytorch_lightning.callbacks import EarlyStopping

from ...dataloader.stgcn import STGCNDataModule
from ...models.stgcn.earlystopping import EarlyStopping as CustomEarlyStopping
from ...models.stgcn.models import STGCNChebGraphConv, STGCNGraphConv
from ...models.stgcn.opt import Lion, Tiger
from dataclasses import dataclass

@dataclass
class Config:
    dataset: str = 'songdo'
    gso_type: str = 'laplacian'
    graph_conv_type: str = 'cheb_graph_conv'
    stblock_num: int = 1
    n_his: int = 12
    n_pred: int = 12
    Kt: int = 3
    lr: float = 0.001
    opt: str = 'adamw'
    weight_decay_rate: float = 0.0005
    step_size: int = 50
    gamma: float = 0.5
    epochs: int = 100
    patience: int = 10
    enable_cuda: bool = True
    device: torch.device = None
    seed: int = 42

class STGCN(pl.LightningModule):
    def __init__(self, config: Config, blocks, n_vertex):
        super().__init__()
        self.config = config
        if config.graph_conv_type == 'cheb_graph_conv':
            self.model = STGCNChebGraphConv(config, blocks, n_vertex)
        else:
            self.model = STGCNGraphConv(config, blocks, n_vertex)
        self.loss = nn.MSELoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).view(len(x), -1)
        loss = self.loss(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).view(len(x), -1)
        loss = self.loss(y_hat, y)
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).view(len(x), -1)
        loss = self.loss(y_hat, y)
        self.log('test_loss', loss)
        return loss

    def configure_optimizers(self):
        if self.config.opt == "adamw":
            optimizer = optim.AdamW(self.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay_rate)
        elif self.config.opt == 'lion':
            optimizer = Lion(self.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay_rate)
        elif self.config.opt == 'tiger':
            optimizer = Tiger(self.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay_rate)
        else:
            raise ValueError(f'ERROR: The {self.config.opt} optimizer is undefined.')

        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.config.step_size, gamma=self.config.gamma)
        return [optimizer], [scheduler]

def set_env(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True