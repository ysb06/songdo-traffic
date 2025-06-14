from datetime import datetime, timedelta
import logging
import os
from typing import List, Optional, Set, Union

import numpy as np
import pandas as pd
from pandas import DateOffset
from tqdm import tqdm

from .interpolation import Interpolator
from .outlier import OutlierProcessor

logger = logging.getLogger(__name__)


class TrafficData:
    @staticmethod
    def import_from_pickle(
        filepath: str, dtype: Optional[Union[str, type]] = float
    ) -> "TrafficData":
        raw: pd.DataFrame = pd.read_pickle(filepath)

        temp = {road_id: {} for road_id in raw["linkID"].unique()}

        # 날짜별로 묶음
        traffic_date_group = raw.groupby("statDate")
        # 날짜별로 로드
        logger.info(f"Loading data from pickle file ({filepath})...")
        with tqdm(total=len(traffic_date_group)) as pbar:
            for date, group in traffic_date_group:
                # 시간별로 로드
                for n in range(24):
                    pbar.set_description(f"{date} {n}h", refresh=True)

                    row_col_key = "hour{:02d}".format(
                        n
                    )  # 읽어들일 열 이름 (ex. hour00, hour01, ...)
                    row_index = datetime.strptime(date, "%Y-%m-%d") + timedelta(
                        hours=n
                    )  # 인덱스로 사용할 날짜
                    for _, row in group.iterrows():
                        temp[row["linkID"]][row_index] = row[row_col_key]

                pbar.update(1)

        data = pd.DataFrame(temp)
        data.sort_index(inplace=True)
        notna_road_ids = data.columns[data.notna().any()].tolist()
        data = data.loc[:, notna_road_ids]

        return TrafficData(data, dtype=dtype)

    @staticmethod
    def import_from_hdf(
        filepath: str,
        key: Optional[str] = None,
        dtype: Optional[Union[str, type]] = float,
    ) -> "TrafficData":
        logger.info(f"Loading data from {filepath}...")
        if key is not None:
            data = pd.read_hdf(filepath, key=key)
        else:
            data = pd.read_hdf(filepath)

        return TrafficData(data, dtype=dtype, path=filepath)

    def __init__(
        self,
        raw: pd.DataFrame,
        dtype: Optional[Union[str, type]] = None,
        freq: Optional[Union[str, DateOffset]] = DateOffset(hours=1),
        path: Optional[str] = None,
    ) -> None:
        self._raw = raw
        self._raw.sort_index(inplace=True)
        if dtype is not None:
            self._raw = self._as_type(self._raw, dtype)
        if freq is None:
            freq = pd.infer_freq(self._raw.index)
            logger.info(f"Inferred frequency: {freq}")
        self._raw = self._raw.asfreq(freq)
        self._verify_data()
        self.reset_data()

        self.path = path

    def _as_type(self, data: pd.DataFrame, dtype: Union[str, type]) -> pd.DataFrame:
        if np.issubdtype(np.dtype(dtype), np.integer):
            if not np.issubdtype(data.dtypes.unique()[0], np.integer):
                logger.warning("Rounding data to integer")
            return data.round().astype(dtype)
        else:
            return data.astype(dtype)

    def _verify_data(self) -> None:
        if not self._raw.index.is_monotonic_increasing:
            raise ValueError("Data not sorted by time")

    def reset_data(self) -> None:
        self.data = self._raw.copy()
        self._sensor_filter = self.data.columns.to_list()
        self.__start_time = self.data.index.min()
        self.__end_time = self.data.index.max()

    @property
    def missings_info(self) -> pd.DataFrame:
        missings = self._raw.isna()
        new_sensor_ids = self._raw.columns.intersection(self._sensor_filter)
        return missings[new_sensor_ids].copy()

    @property
    def sensor_filter(self) -> list[str]:
        return self._sensor_filter

    @sensor_filter.setter
    def sensor_filter(self, sensor_ids: list[str]) -> None:
        new_sensor_ids = self._raw.columns.intersection(sensor_ids)
        self._sensor_filter = new_sensor_ids
        self.data = self._raw[new_sensor_ids].copy()

    @property
    def start_time(self) -> pd.Timestamp:
        return self.__start_time

    @start_time.setter
    def start_time(self, start_time: Union[str, pd.Timestamp]) -> None:
        self.__start_time = pd.Timestamp(start_time)
        self.data = self.data.loc[self.__start_time :]

    @property
    def end_time(self) -> pd.Timestamp:
        return self.__end_time

    @end_time.setter
    def end_time(self, end_time: Union[str, pd.Timestamp]) -> None:
        self.__end_time = pd.Timestamp(end_time)
        self.data = self.data.loc[: self.__end_time]

    @property
    def original_data(self) -> pd.DataFrame:
        return self._raw.copy()[self._sensor_filter].loc[
            self.__start_time : self.__end_time
        ]

    def remove_outliers(
        self, processor: Union[OutlierProcessor, List[OutlierProcessor]]
    ) -> Set[str]:
        if not isinstance(processor, list):
            processor = [processor]
        logger.info("Process Outlier for METR-IMC Traffic data...")
        failed_set = set()
        for proc in processor:
            logger.info(f"Processing with {proc.__class__.__name__}...")
            self.data = proc.process(self.data)
            failed_set.update(proc.failed_list)
        logger.info("Processing Complete")
        # 언젠가 코드를 리팩토링 해야할 것 같음
        # Outlier 처리나 Interpolation은 외부로 빼는 것이 좋을 것 같음
        return failed_set

    def interpolate(self, interpolator: Interpolator):
        logger.info("Interpolating METR-IMC Traffic data...")
        self.data = interpolator._interpolate(self.data)
        logger.info("Interplating Complete")

    def to_hdf(self, filepath: str, key: str = "data") -> None:
        logger.info(f"Saving data to {filepath}...")
        self.data.to_hdf(filepath, key=key)
        logger.info(f"Saving Complete...{self.data.shape}")

    def to_excel(self, filepath: str) -> None:
        logger.info(f"Saving data to {filepath}...")
        self.data.to_excel(filepath)
        logger.info(f"Saving Complete...{self.data.shape}")
