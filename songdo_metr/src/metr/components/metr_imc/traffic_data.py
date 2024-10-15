import logging
import os
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from pandas.tseries.offsets import DateOffset

from .interpolation import Interpolator
from .outlier import OutlierProcessor

logger = logging.getLogger(__name__)


class TrafficData:
    @staticmethod
    def import_from_hdf(
        filepath: str,
        key: Optional[str] = None,
        dtype: Optional[Union[str, type]] = None,
    ) -> "TrafficData":
        if key is not None:
            data = pd.read_hdf(filepath, key=key)
        else:
            data = pd.read_hdf(filepath)
        return TrafficData(data, dtype=dtype)

    def __init__(
        self, raw: pd.DataFrame, dtype: Optional[Union[str, type]] = None
    ) -> None:
        self._raw = raw
        self._raw.sort_index(inplace=True)
        if dtype is not None:
            self._raw = self._as_type(self._raw, dtype)
        self._raw = self._raw.asfreq(pd.infer_freq(self._raw.index))
        self._verify_data()
        self.reset_data()
    
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
    def is_missing_values(self) -> pd.DataFrame:
        missings = self._raw.isna()
        new_sensor_ids = self._raw.columns.intersection(self._sensor_filter)
        return missings[new_sensor_ids].copy()

    @property
    def sensor_filter(self) -> list[str]:
        return self._sensor_filter

    @sensor_filter.setter
    def sensor_filter(self, sensor_ids: list[str]) -> None:
        missing_sensors = set(sensor_ids) - set(self._raw.columns)
        if missing_sensors:
            logger.warning(
                f"The following sensors do not exist in the data:\r\n{', '.join(missing_sensors)}"
            )
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

    def remove_outliers(self, processor: Union[OutlierProcessor, List[OutlierProcessor]]) -> None:
        if not isinstance(processor, list):
            processor = [processor]
        logger.info("Process Outlier for METR-IMC Traffic data...")
        for proc in processor:
            print(f"Processing with {proc.__class__.__name__}...")
            self.data = proc.process(self.data)
        logger.info("Processing Complete")

    def interpolate(self, interpolator: Interpolator) -> pd.DataFrame:
        logger.info("Interpolating METR-IMC Traffic data...")
        self.data = interpolator.interpolate(self.data)
        logger.info("Interplating Complete")

    def remove_weird_zero(self) -> None:
        """Remove zeros around missing values"""

        def extend_nans_around_zeros(series: pd.Series) -> pd.Series:
            series = series.copy()
            nan_indices = series[series.isna()].index

            for idx in nan_indices:
                idx_pos = series.index.get_loc(idx)

                i = idx_pos - 1
                while i >= 0 and series.iat[i] == 0:
                    series.iat[i] = np.nan
                    i -= 1

                i = idx_pos + 1
                while i < len(series) and series.iat[i] == 0:
                    series.iat[i] = np.nan
                    i += 1

            return series

        self.data = self.data.apply(extend_nans_around_zeros)

    def to_hdf(self, filepath: str, key: Optional[str] = None) -> None:
        logger.info(f"Saving data to {filepath}...")
        if key is not None:
            self.data.to_hdf(filepath, key=key)
        else:
            self.data.to_hdf(filepath, key="data")
        logger.info("Saving Complete")

    def to_excel(self, filepath: str) -> None:
        logger.info(f"Saving data to {filepath}...")
        self.data.to_excel(filepath)
        logger.info("Saving Complete")
