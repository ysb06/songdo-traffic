from dataclasses import dataclass
from typing import Literal


@dataclass
class STGCNConfig:
    # Dataset
    dataset: str = "metr-la"
    n_his: int = 12
    n_pred: int = 3
    time_intvl: int = 5

    # Model
    Kt: int = 3
    stblock_num: int = 2
    act_func: Literal["glu", "gtu"] = "glu"
    Ks: Literal[3, 2] = 3
    graph_conv_type: Literal["cheb_graph_conv", "graph_conv"] = "cheb_graph_conv"
    gso_type: Literal[
        "sym_norm_lap", "rw_norm_lap", "sym_renorm_adj", "rw_renorm_adj"
    ] = "sym_norm_lap"
    enable_bias: bool = True
    droprate: float = 0.5

    # Training
    seed: int = 42
    lr: float = 0.001
    weight_decay_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 1000
    patience: int = 10

    # Optimizer
    optimizer: Literal["adamw", "lion", "tiger"] = "lion"
    step_size: int = 10
    gamma: float = 0.95

    # Hardware
    device: str = "cpu"


def load_config(config_path: str) -> STGCNConfig:
    import yaml

    with open(config_path, "r") as file:
        config_dict = yaml.safe_load(file)
    return STGCNConfig(**config_dict)
