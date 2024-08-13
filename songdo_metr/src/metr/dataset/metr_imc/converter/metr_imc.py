from typing import List, Optional, Tuple
import geopandas as gpd
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime, timedelta
import logging
import os

from metr.dataset.interpolator import Interpolator

logger = logging.getLogger(__name__)


class MetrImcTrafficData:
    """metr-imc.h5"""

    def __init__(self, data: Optional[pd.DataFrame] = None) -> None:
        self.raw = data
        self.data = self.raw.copy() if self.raw is not None else None

    @property
    def data_exists(self) -> bool:
        return self.data is not None

    def import_from_imcrts(self, data_path: str):
        traffic_data_raw = pd.read_pickle(data_path)
        converted_dict = self.__convert_raw_data(traffic_data_raw)
        self.raw = pd.DataFrame(converted_dict, dtype=np.float64)
        self.data = self.raw.copy()

    def __convert_raw_data(self, raw_traffic_data: pd.DataFrame):
        temp = {road_id: {} for road_id in raw_traffic_data["linkID"].unique()}

        traffic_date_group = raw_traffic_data.groupby("statDate")
        for date, group in tqdm(traffic_date_group):
            for n in range(24):
                row_col_key = "hour{:02d}".format(
                    n
                )  # 읽어들일 열 이름 (ex. hour00, hour01, ...)
                row_index = datetime.strptime(date, "%Y-%m-%d") + timedelta(
                    hours=n
                )  # 인덱스로 사용할 datetime 형식
                for _, row in group.iterrows():
                    if row["linkID"] in temp:
                        temp[row["linkID"]][row_index] = row[row_col_key]

        return temp

    def select_columns(self, column_list: Optional[List[str]]):
        self.data = self.raw.copy()
        if column_list is not None:
            self.data = self.data[column_list]

    def select_period(
        self,
        idx_list: Optional[List[int]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ):
        if idx_list is not None:
            self.data = self.data.loc[idx_list]
        else:
            if start_date is not None and end_date is not None:
                self.data = self.data.loc[start_date:end_date]
            elif start_date is not None:
                self.data = self.data.loc[start_date:]
            elif end_date is not None:
                self.data = self.data.loc[:end_date]

    def interpolate(self, interpolator: Interpolator) -> pd.DataFrame:
        logger.info("Interpolating METR-IMC data...")
        self.data = interpolator.interpolate(self.data)
        logger.info("Interplating Complete")

    def to_hdf(self, dir_path: str, filename: str = "metr-imc.h5") -> None:
        logger.info(f"Saving METR-IMC data to {os.path.join(dir_path, filename)}...")
        self.data.to_hdf(os.path.join(dir_path, filename), key="data")
        logger.info("Complete")

    def to_excel(self, dir_path: str, filename: str = "metr-imc.xlsx") -> None:
        logger.info(f"Saving METR-IMC data to {os.path.join(dir_path, filename)}...")
        self.data.to_excel(os.path.join(dir_path, filename))
        logger.info("Complete")


class MetrImc:
    """metr-imc.h5"""

    def __init__(
        self,
        traffic_data: pd.DataFrame,
        road_data: gpd.GeoDataFrame,
        remove_empty: bool = True,
    ) -> None:
        road_ids = list(set(road_data["LINK_ID"]) & set(traffic_data["linkID"]))

        logger.info("Converting traffic data to METR format...")
        converted_dict, missing_links = self.__convert_raw_data(traffic_data, road_ids)
        # Todo: converted_dict가 제대로 변환되었는지 확인 필요
        # 정상적이라면 missing_links는 빈 리스트가 되어야 함, 이미 road_ids에서 처리를 했으므로...

        if missing_links:
            logger.warning(
                f"There is some traffic data whose road is not in the Standard Node-Link: {len(missing_links)}"
            )
            print(missing_links[:5] + ["..."])

        self.data = pd.DataFrame(converted_dict, dtype=np.float32)
        logger.info(f"Data Size: {self.data.shape[0]}")
        if remove_empty:
            self.data = self.data.loc[:, self.notna_road_ids]

        logger.info("Complete")

    @property
    def road_ids(self):
        return self.data.columns.tolist()

    @property
    def notna_road_ids(self):
        return self.data.columns[self.data.notna().any()].tolist()

    def __convert_raw_data(
        self, raw_traffic_data: pd.DataFrame, target_road_ids: List[str]
    ):
        temp = {road_id: {} for road_id in target_road_ids}
        missing_links = []  # For logging and debugging

        # 날짜별로 묶음
        traffic_date_group = raw_traffic_data.groupby("statDate")
        # 날짜별로 로드
        for date, group in tqdm(traffic_date_group):
            # 시간별로 로드
            for n in range(24):
                row_col_key = "hour{:02d}".format(
                    n
                )  # 읽어들일 열 이름 (ex. hour00, hour01, ...)
                row_index = datetime.strptime(date, "%Y-%m-%d") + timedelta(
                    hours=n
                )  # 인덱스로 사용할 날짜
                for _, row in group.iterrows():
                    if row["linkID"] in temp:
                        temp[row["linkID"]][row_index] = row[row_col_key]
                    else:
                        missing_links.append(row["linkID"])

        return temp, missing_links

    def select_roads_with_data(
        self, road_data: gpd.GeoDataFrame
    ) -> Tuple[gpd.GeoDataFrame, pd.DataFrame]:
        """도로 링크 데이터에서 교통량 데이터가 존재하는 링크만 추출

        Args:
            road_data (gpd.GeoDataFrame): 도로 링크 데이터

        Returns:
            Tuple[gpd.GeoDataFrame, pd.DataFrame]: 필터링된 도로 링크 데이터, 필터링된 교통량 데이터
        """
        index_values = self.notna_road_ids

        return (
            road_data[road_data["LINK_ID"].isin(index_values)],
            self.data.loc[:, index_values],
        )

    def to_hdf(self, dir_path: str, filename: str = "metr-imc.h5") -> None:
        logger.info(f"Saving METR data to {os.path.join(dir_path, filename)}...")
        self.data.to_hdf(os.path.join(dir_path, filename), key="data")
        logger.info("Complete")

    def to_excel(self, dir_path: str, filename: str = "metr-imc.xlsx") -> None:
        logger.info(f"Saving METR data to {os.path.join(dir_path, filename)}...")
        self.data.to_excel(os.path.join(dir_path, filename))
        logger.info("Complete")
