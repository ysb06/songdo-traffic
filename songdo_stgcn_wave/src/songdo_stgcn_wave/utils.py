from typing import Dict, Optional, Union
import torch
import random
from dataclasses import dataclass
import yaml
import numpy as np


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
    train_ratio: Optional[float] = 0.7
    valid_ratio: Optional[float] = 0.1
    drop_rate: Optional[float] = 0.0
    scheduler: Optional[Dict[str, Union[int, float]]] = {"step_size": 5, "gamma": 0.7}


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

def fix_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False