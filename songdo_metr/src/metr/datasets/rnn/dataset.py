import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, Subset

logger = logging.getLogger(__name__)

TrafficDataType = Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex, pd.Timestamp]
TrafficMultiSensorDataType = Tuple[
    np.ndarray, np.ndarray, pd.DatetimeIndex, pd.Timestamp, str
]
TrafficMultiSensorWithMissingDataType = Tuple[
    np.ndarray, np.ndarray, pd.DatetimeIndex, pd.Timestamp, str, bool
]


class TrafficCoreDataset(Dataset):
    def __init__(
        self,
        data: np.ndarray,
        seq_length: int = 24,
        allow_nan: bool = False,
    ):
        super().__init__()
        if len(data) < seq_length:
            raise ValueError("Data length should be larger than seq_length")

        self.seq_length = seq_length
        self.data = data.reshape(-1, 1)
        self.scaled_data = self.data

        self.cursors: List[int] = []
        time_len = len(self.data)
        # 1단위(시간)씩 슬라이딩하면서 커서를 생성
        if not allow_nan:
            # NaN이 있는 경우, NaN이 없는 구간만 커서를 저장
            isnan_arr = np.isnan(self.data).reshape(-1)
            cumsum = np.cumsum(isnan_arr, dtype=np.int32)
            cumsum = np.insert(cumsum, 0, 0)
            all_i = np.arange(time_len - seq_length)
            x_nan_count = cumsum[all_i + seq_length] - cumsum[all_i]
            y_is_nan = isnan_arr[all_i + seq_length]
            valid_mask = (x_nan_count == 0) & (~y_is_nan)
            valid_i = all_i[valid_mask]
            self.cursors = valid_i.tolist()
        else:
            # NaN에 상관없이 모든 데이터의 커서를 저장
            self.cursors = list(range(time_len - seq_length))

    def __len__(self) -> int:
        return len(self.cursors)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray, List[int], int]:
        # cursors에서 실제 데이터 인덱스를 가져옴
        cursor = self.cursors[index]
        x = self.scaled_data[cursor : cursor + self.seq_length]
        y = self.scaled_data[cursor + self.seq_length]

        x_idxs = list(range(cursor, cursor + self.seq_length))
        y_idx = cursor + self.seq_length

        return x, y, x_idxs, y_idx

    def apply_scaler(self, scaler: MinMaxScaler):
        self.scaled_data = scaler.transform(self.data)


class TrafficDataset(TrafficCoreDataset):
    def __init__(
        self,
        data: pd.Series,
        seq_length: int = 24,
        allow_nan: bool = False,
    ):
        super().__init__(data.to_numpy(), seq_length, allow_nan)
        self.data_df = data

    def __getitem__(self, index: int) -> TrafficDataType:
        """원래의 numpy기반 데이터를 반환하고 pandas의 시간 인덱스까지 반환.

        Args:
            index (int): 커서 위치

        Returns:
            TrafficDataType: x, y 데이터 및 해당 데이터의 index
        """
        x, y, x_idxs, y_idx = super().__getitem__(index)

        x_time_indices = self.data_df.index[x_idxs]  # x_idxs = range(i, i+seq_length)
        y_time_index = self.data_df.index[y_idx]  # y_idx = i + seq_length

        return x, y, x_time_indices, y_time_index

    @property
    def name(self) -> str:
        return self.data_df.name

    def get_subset(
        self,
        start_datetime: Optional[str] = None,
        end_datetime: Optional[str] = None,
    ) -> Subset:
        if start_datetime is None:
            start_ts = self.data_df.index[0]
        else:
            start_ts = pd.to_datetime(start_datetime)

        if end_datetime is None:
            end_ts = self.data_df.index[-1]
        else:
            end_ts = pd.to_datetime(end_datetime)

        label_times = self.data_df.index[[c + self.seq_length for c in self.cursors]]
        mask = (label_times >= start_ts) & (
            label_times <= end_ts - pd.Timedelta(hours=self.seq_length)
        )
        valid_indices = np.where(mask)[0].tolist()

        return Subset(self, valid_indices)


class TrafficMultiSensorDataset(Dataset):
    """다중 센서 데이터를 처리하는 데이터셋 클래스.

    DataFrame의 각 컬럼이 서로 다른 센서를 나타내는 경우 사용.
    각 샘플마다 센서 이름 정보도 함께 반환.
    missing_mask가 제공되면 y값이 원래 결측치였는지 여부도 반환.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        seq_length: int = 24,
        allow_nan: bool = False,
        missing_mask: Optional[pd.DataFrame] = None,
    ):
        super().__init__()
        self.data_df = data
        self.seq_length = seq_length
        self.sensor_names = list(data.columns)
        self.missing_mask = missing_mask

        # 각 센서별로 TrafficDataset을 생성
        self.sensor_datasets: dict[str, TrafficDataset] = {}
        self.sensor_sample_counts: dict[str, int] = {}

        for sensor_name in self.sensor_names:
            sensor_series = data[sensor_name]
            sensor_dataset = TrafficDataset(sensor_series, seq_length, allow_nan)
            self.sensor_datasets[sensor_name] = sensor_dataset
            self.sensor_sample_counts[sensor_name] = len(sensor_dataset)

        # 센서별 샘플 인덱스 매핑 생성
        self._create_index_mapping()

    def _create_index_mapping(self):
        """전체 인덱스를 (센서명, 센서내 인덱스)로 매핑하는 테이블 생성"""
        self.index_mapping = []

        for sensor_name in self.sensor_names:
            sensor_sample_count = self.sensor_sample_counts[sensor_name]
            for i in range(sensor_sample_count):
                self.index_mapping.append((sensor_name, i))

    def __len__(self) -> int:
        return sum(self.sensor_sample_counts.values())

    def __getitem__(self, index: int) -> TrafficMultiSensorDataType:
        sensor_name, sensor_index = self.index_mapping[index]
        sensor_dataset = self.sensor_datasets[sensor_name]

        x, y, x_time_indices, y_time_index = sensor_dataset[sensor_index]

        # missing_mask가 있으면 y값이 원래 결측치였는지 확인
        if self.missing_mask is not None:
            y_is_missing = bool(self.missing_mask.loc[y_time_index, sensor_name])
            return x, y, x_time_indices, y_time_index, sensor_name, y_is_missing

        return x, y, x_time_indices, y_time_index, sensor_name
