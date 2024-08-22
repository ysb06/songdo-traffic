from typing import Optional
import torch
import gc
import os
import random
from datetime import datetime
from dataclasses import dataclass
import yaml

@dataclass
class HyperParams:
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
    # Extended Args
    adj_mx_filepath: Optional[str] = None

def get_auto_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    return device

def get_config(config_path: str) -> HyperParams:
    with open(config_path, "r") as f:
        config_raw = yaml.load(f, Loader=yaml.FullLoader)

    return HyperParams(**config_raw)
