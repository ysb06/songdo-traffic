import logging
from typing import Callable, List, Tuple

import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from metr.datasets.rnn.dataset import TrafficDataset, TrafficDataType

logger = logging.getLogger(__name__)

CollateFnInput = List[TrafficDataType]


def collate_all(
    batch: CollateFnInput,
) -> Tuple[Tensor, Tensor, List[pd.DatetimeIndex], List[pd.Timestamp]]:
    xs = [item[0] for item in batch]
    ys = [item[1] for item in batch]
    x_time_indices = [item[2] for item in batch]
    y_time_indices = [item[3] for item in batch]

    xs_t = [torch.from_numpy(x).float() for x in xs]
    ys_t = [torch.from_numpy(y).float() for y in ys]

    xs_t = torch.stack(xs_t, dim=0)
    ys_t = torch.stack(ys_t, dim=0)

    return xs_t, ys_t, x_time_indices, y_time_indices


def collate_simple(batch: CollateFnInput) -> Tuple[Tensor, Tensor]:
    xs = [item[0] for item in batch]
    ys = [item[1] for item in batch]

    xs_t = [torch.from_numpy(x).float() for x in xs]
    ys_t = [torch.from_numpy(y).float() for y in ys]

    xs_t = torch.stack(xs_t, dim=0)
    ys_t = torch.stack(ys_t, dim=0)

    return xs_t, ys_t


def get_dataset(
    dataset: TrafficDataset,
    batch_size: int = 64,
    shuffle: bool = False,
    num_workers: int = 0,
    collate_fn: Callable[[CollateFnInput], Tuple] = collate_simple,
):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
