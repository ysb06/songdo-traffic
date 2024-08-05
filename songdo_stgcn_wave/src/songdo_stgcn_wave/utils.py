import torch
import gc
import os
import random
from datetime import datetime
from dataclasses import dataclass

@dataclass
class Config:
    lr: float
    disablecuda: bool
    batch_size: int
    epochs: int
    num_layers: int
    window: int
    dataset_name: str
    sensorsfilepath: str
    disfilepath: str
    tsfilepath: str
    savemodelpath: str
    pred_len: int
    control_str: str
    channels: list
    seed: int

def get_auto_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    return device
