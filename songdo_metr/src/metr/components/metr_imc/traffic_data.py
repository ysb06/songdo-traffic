import logging
import os
from typing import Optional, Union

import pandas as pd
from pandas.tseries.offsets import DateOffset

from .interpolation import Interpolator
from .outlier import OutlierProcessor

logger = logging.getLogger(__name__)


class TrafficData:
    @staticmethod
    def import_from_hdf(filepath: str, key: Optional[str] = None) -> "TrafficData":
        if key is not None:
            data = pd.read_hdf(filepath, key=key)
        else:
            data = pd.read_hdf(filepath)
        return TrafficData(data)

    def __init__(self, raw: pd.DataFrame) -> None:
        self.__raw = raw
        self.__raw.sort_index(inplace=True)
        self.__raw = self.__raw.asfreq(pd.infer_freq(self.__raw.index))
        self.missings = self.__raw.isna()
        self.__verify_data()
        self.reset_data()

    def __verify_data(self) -> None:
        if not self.__raw.index.is_monotonic_increasing:
            raise ValueError("Data not sorted by time")
        else:
            logger.info(f"Data sorted by time: {pd.infer_freq(self.__raw.index)}")

    def reset_data(self) -> None:
        self.data = self.__raw.copy()
        self.__sensor_filter = self.data.columns.to_list()
        self.__start_time = self.data.index.min()
        self.__end_time = self.data.index.max()

    @property
    def sensor_filter(self) -> list[str]:
        return self.__sensor_filter

    @sensor_filter.setter
    def sensor_filter(self, sensor_ids: list[str]) -> None:
        self.__sensor_filter = sensor_ids
        self.data = self.data[sensor_ids]

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

    def delete_outliers(self, processor: OutlierProcessor) -> None:
        logger.info("Process Outlier for METR-IMC Traffic data...")
        self.data = processor.process(self.data)
        logger.info("Processing Complete")

    def interpolate(self, interpolator: Interpolator) -> pd.DataFrame:
        logger.info("Interpolating METR-IMC Traffic data...")
        self.data = interpolator.interpolate(self.data)
        logger.info("Interplating Complete")

    def to_hdf(self, filepath: str, key: Optional[str] = None) -> None:
        logger.info(f"Saving data to {filepath}...")
        if key is not None:
            self.data.to_hdf(filepath, key=key)
        else:
            self.data.to_hdf(filepath)
        logger.info("Saving Complete")

    def export_all(self, dir_path: str, filename_prefix: str = "metr-imc") -> None:
        logger.info(f"Exporting all data to {dir_path}...")
        traffic_data_path = os.path.join(dir_path, f"{filename_prefix}.h5")
        missing_data_path = os.path.join(dir_path, f"{filename_prefix}-missing.h5")
        raw_data_path = os.path.join(dir_path, f"{filename_prefix}-raw.h5")

        self.data.to_hdf(traffic_data_path)
        self.missings.to_hdf(missing_data_path)
        self.__raw.to_hdf(raw_data_path)
        logger.info("Exporting Complete")
