import torch
import torch.nn as nn
import lightning as L
from typing import Optional, Tuple
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from .model import STGCNGraphConv


class STGCNLightningModule(L.LightningModule):
    def __init__(self):
        super().__init__()