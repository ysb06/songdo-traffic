import argparse
import gc
import os
import random
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import dgl
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
import wandb
from sklearn.preprocessing import StandardScaler

from .utils import Config, get_auto_device


def train(config: Config):
    run_name = f"{config.dataset_name}_STGCN_WAVE_{datetime.now().strftime('%y%m%d_%H%M%S')}"
    training_divice = get_auto_device()
    # wandb.init(project='METR-IMC', name=run_name, config=asdict(config))
    # wandb.config.update({"device": str(training_divice)})
    
    print(run_name, training_divice)