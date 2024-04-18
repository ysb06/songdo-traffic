from typing import List, Tuple
import geopandas as gpd
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class MetrImc:
    """metr-imc.h5"""

    def __init__(
        self,
        traffic_data: pd.DataFrame,
        road_data: gpd.GeoDataFrame,
        remove_empty: bool = True,
    ) -> None:
        road_ids = list(set(road_data["LINK_ID"]) & set(traffic_data["linkID"]))

        logger.info("Converting traffic data to METR-IMC format...")
        converted_dict, missing_links = self.__convert_raw_data(traffic_data, road_ids)

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

        traffic_date_group = raw_traffic_data.groupby("statDate")
        for date, group in tqdm(traffic_date_group):
            for n in range(24):
                row_col_key = "hour{:02d}".format(n)  # 읽어들일 열 이름
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

    def to_hdf(self, dir_path: str) -> None:
        logger.info(f"Saving METR-IMC data to {dir_path}/metr_imc.h5...")
        self.data.to_hdf(f"{dir_path}/metr-imc.h5", key="data")
        logger.info("Complete")

    def to_excel(self, dir_path: str) -> None:
        logger.info(f"Saving METR-IMC data to {dir_path}/metr_imc.xlsx")
        self.data.to_excel(f"{dir_path}/metr-imc.xlsx")
        logger.info("Complete")
