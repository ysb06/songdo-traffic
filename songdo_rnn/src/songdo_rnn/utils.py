import random
from typing import Sequence, Union

import numpy as np
import torch


def get_auto_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def symmetric_mean_absolute_percentage_error(
    y_true: Union[np.ndarray, Sequence[float]],
    y_pred: Union[np.ndarray, Sequence[float]],
    epsilon: float = 1e-8,
) -> float:
    y_true_arr = np.array(y_true, dtype=float)
    y_pred_arr = np.array(y_pred, dtype=float)

    numerator = np.abs(y_true_arr - y_pred_arr)
    denominator = (np.abs(y_true_arr) + np.abs(y_pred_arr)) / 2

    # 분모가 0이거나 매우 작아지는 것을 방지
    denominator = np.where(denominator == 0, epsilon, denominator)

    smape_val = np.mean(numerator / denominator) * 100.0

    return smape_val.item()


def fix_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False