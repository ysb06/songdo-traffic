"""
RNN model dataset preparation and testing utilities for traffic prediction
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
import pandas as pd
import torch
import numpy as np
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from metr.datasets.rnn.datamodule import TrafficDataModule
from metr_val.model import BasicRNN
from metr_val import PATH_CONF

logger = logging.getLogger(__name__)


datamodule = TrafficDataModule(PATH_CONF.dataset.metr_imc)
