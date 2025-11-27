from datetime import datetime, timedelta
import logging
import os
from typing import List, Optional, Set, Tuple, Union

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
        freq: Optional[str] = "h",
        path: Optional[str] = None,
    ) -> None:
        raw.sort_index(inplace=True)
        if dtype is not None:
            raw = self._as_type(raw, dtype)
        if freq is None:
            freq = pd.infer_freq(raw.index)
            logger.info(f"Inferred frequency: {freq}")
        raw = raw.asfreq(freq)
        self._verify_data(raw)
        self.data = raw

        self.path = path

    def _as_type(self, data: pd.DataFrame, dtype: Union[str, type]) -> pd.DataFrame:
        if np.issubdtype(np.dtype(dtype), np.integer):
            if not np.issubdtype(data.dtypes.unique()[0], np.integer):
                logger.warning("Rounding data to integer")
            return data.round().astype(dtype)
        else:
            return data.astype(dtype)

    def _verify_data(self, raw: pd.DataFrame) -> None:
        if not raw.index.is_monotonic_increasing:
            raise ValueError("Data not sorted by time")
    
    def split(self, split_date: pd.Timestamp) -> Tuple["TrafficData", "TrafficData"]:
        """Split the data into two TrafficData instances at the specified date.

        Args:
            split_date: The date to split the data on.

        Returns:
            A tuple containing two TrafficData instances: (data_before_split, data_after_split)
        """
        data_before = self.data[self.data.index < split_date]
        data_after = self.data[self.data.index >= split_date]
        
        return TrafficData(data_before), TrafficData(data_after)

    def to_hdf(self, filepath: str, key: str = "data") -> None:
        logger.info(f"Saving data to {filepath}...")
        self.data.to_hdf(filepath, key=key)
        logger.info(f"Saving Complete...{self.data.shape}")

    def to_excel(self, filepath: str) -> None:
        logger.info(f"Saving data to {filepath}...")
        self.data.to_excel(filepath)
        logger.info(f"Saving Complete...{self.data.shape}")


def get_raw(path: str) -> TrafficData:
    """
    Load raw traffic data from the specified path.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    return TrafficData.import_from_hdf(path)
