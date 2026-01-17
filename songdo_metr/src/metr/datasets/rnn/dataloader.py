import logging
from typing import Callable, List, Tuple

import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from metr.datasets.rnn.dataset import (
    TrafficDataset,
    TrafficDataType,
    TrafficMultiSensorDataType,
    TrafficMultiSensorWithMissingDataType,
)

logger = logging.getLogger(__name__)

CollateFnInput = List[TrafficDataType]
MultiSensorCollateFnInput = List[TrafficMultiSensorDataType]
MultiSensorWithMissingCollateFnInput = List[TrafficMultiSensorWithMissingDataType]


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


def collate_multi_sensor(
    batch: MultiSensorCollateFnInput,
) -> Tuple[Tensor, Tensor, List[pd.DatetimeIndex], List[pd.Timestamp], List[str]]:
    """TrafficMultiSensorDataset용 collate 함수. collate_all과 동일한 형태에 센서 이름 추가."""
    xs = [item[0] for item in batch]
    ys = [item[1] for item in batch]
    x_time_indices = [item[2] for item in batch]
    y_time_indices = [item[3] for item in batch]
    sensor_names = [item[4] for item in batch]

    xs_t = [torch.from_numpy(x).float() for x in xs]
    ys_t = [torch.from_numpy(y).float() for y in ys]

    xs_t = torch.stack(xs_t, dim=0)
    ys_t = torch.stack(ys_t, dim=0)

    return xs_t, ys_t, x_time_indices, y_time_indices, sensor_names


def collate_multi_sensor_simple(
    batch: MultiSensorCollateFnInput,
) -> Tuple[Tensor, Tensor, List[str]]:
    """TrafficMultiSensorDataset용 간단한 collate 함수. 데이터와 센서 이름만 반환."""
    xs = [item[0] for item in batch]
    ys = [item[1] for item in batch]

    xs_t = [torch.from_numpy(x).float() for x in xs]
    ys_t = [torch.from_numpy(y).float() for y in ys]

    xs_t = torch.stack(xs_t, dim=0)
    ys_t = torch.stack(ys_t, dim=0)

    return xs_t, ys_t


def collate_multi_sensor_with_missing(
    batch: MultiSensorWithMissingCollateFnInput,
) -> Tuple[Tensor, Tensor, List[pd.DatetimeIndex], List[pd.Timestamp], List[str], List[bool]]:
    """TrafficMultiSensorDataset용 테스트 collate 함수. missing 정보 포함."""
    xs = [item[0] for item in batch]
    ys = [item[1] for item in batch]
    x_time_indices = [item[2] for item in batch]
    y_time_indices = [item[3] for item in batch]
    sensor_names = [item[4] for item in batch]
    y_is_missing_list = [item[5] for item in batch]

    xs_t = [torch.from_numpy(x).float() for x in xs]
    ys_t = [torch.from_numpy(y).float() for y in ys]

    xs_t = torch.stack(xs_t, dim=0)
    ys_t = torch.stack(ys_t, dim=0)

    return xs_t, ys_t, x_time_indices, y_time_indices, sensor_names, y_is_missing_list


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
