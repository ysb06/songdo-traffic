from typing import Dict, List, Optional, Union
import torch
import random
from dataclasses import dataclass, field
import yaml
import numpy as np


@dataclass
class HyperParams:
    p_max: int
    d_max: int
    q_max: int
    training_data_ratio: float
    traffic_training_data_path: str
    traffic_test_data_path: str
    output_root_dir: str
    info: Dict[str, List[str]]


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