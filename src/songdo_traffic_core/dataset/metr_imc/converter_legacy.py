# 여기 코드는 converter 모듈이 모두 완성되고 삭제할 것

from typing import List, Tuple
import geopandas as gpd
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


def calculate_mean_coords(geometry):
    x_coords = [point[0] for point in geometry.coords]
    y_coords = [point[1] for point in geometry.coords]
    mean_x = np.mean(x_coords)
    mean_y = np.mean(y_coords)
    return pd.Series([mean_x, mean_y], index=["longitude", "latitude"])


class GraphSensorLocations:
    def __init__(self, road_data: gpd.GeoDataFrame) -> None:
        road_data = road_data.to_crs(epsg=4326)

        self.result = pd.DataFrame(columns=["sensor_id", "latitude", "longitude"])
        self.result["sensor_id"] = road_data["LINK_ID"]
        self.result[["longitude", "latitude"]] = road_data["geometry"].apply(
            calculate_mean_coords
        )

    def to_csv(self, dir_path: str) -> None:
        logger.info(
            f"Saving sensor locations to {dir_path}/graph_sensor_locations.csv..."
        )
        self.result.to_csv(
            f"{dir_path}/graph_sensor_locations.csv", index_label="index"
        )
        logger.info("Complete")


class MetrImc:
    """metr-imc.h5"""

    def __init__(self, traffic_data: pd.DataFrame, road_data: gpd.GeoDataFrame) -> None:
        road_ids = list(set(road_data["LINK_ID"]) & set(traffic_data["linkID"]))
        temp = {road_id: {} for road_id in road_ids}

        logger.info("Converting traffic data to METR-IMC format...")
        self.missing_links = []
        traffic_date_group = traffic_data.groupby("statDate")
        for date, group in tqdm(traffic_date_group):
            for n in range(24):
                row_col_key = "hour{:02d}".format(n)
                row_index = datetime.strptime(date, "%Y-%m-%d") + timedelta(hours=n)
                for _, row in group.iterrows():
                    if row["linkID"] in temp:
                        temp[row["linkID"]][row_index] = row[row_col_key]
                    else:
                        self.missing_links.append(row["linkID"])

        self.data = pd.DataFrame(temp, dtype=np.float32)
        if self.missing_links:
            logger.warning(
                f"There is some traffic data whose road is not in the Standard Node-Link: {len(self.missing_links)}"
            )
            print(self.missing_links[:5] + ["..."])
        logger.info("Complete")

    def to_hdf(self, dir_path: str) -> None:
        logger.info(f"Saving METR-IMC data to {dir_path}/metr_imc.h5...")
        self.data.to_hdf(f"{dir_path}/metr_imc.h5", key="data")
        logger.info("Complete")

    def to_excel(self, dir_path: str) -> None:
        logger.info(f"Saving METR-IMC data to {dir_path}/metr_imc.xlsx")
        self.data.to_excel(f"{dir_path}/metr_imc.xlsx")
        logger.info("Complete")

    def select_roads_with_data(
        self, road_data: gpd.GeoDataFrame
    ) -> Tuple[gpd.GeoDataFrame, pd.DataFrame]:
        """도로 링크 데이터에서 교통량 데이터가 존재하는 링크만 추출

        Args:
            road_data (gpd.GeoDataFrame): 도로 링크 데이터

        Returns:
            Tuple[gpd.GeoDataFrame, pd.DataFrame]: 필터링된 도로 링크 데이터, 필터링된 교통량 데이터
        """
        index_values = self.data.columns[self.data.notna().any()].tolist()

        return (
            road_data[road_data["LINK_ID"].isin(index_values)],
            self.data.loc[:, index_values],
        )


class MetrIds:
    def __init__(self, traffic_data: pd.DataFrame) -> None:
        self.ids: List[str] = traffic_data.columns.tolist()

    def to_txt(self, dir_path: str) -> None:
        logger.info(f"Saving METR-IMC IDs to {dir_path}/metr_ids.txt...")
        ids_str = ",".join(str(id) for id in self.ids)
        with open(f"{dir_path}/metr_ids.txt", "w") as file:
            file.write(ids_str)
        logger.info("Complete")

    def to_list(self) -> List[str]:
        return self.ids


class DistancesImc:
    """distances_imc_2024.csv"""

    def __init__(
        self,
        road_data: gpd.GeoDataFrame,  # 표준노드링크 링크 데이터
        distance_limit: float = 2000,  # m 단위
    ) -> None:
        road_data = road_data.to_crs(epsg=5186)
        self.data = pd.DataFrame([])

    def __generate(self, road_data: gpd.GeoDataFrame, distance_limit: float):
        # 깊이 기반 탐색하고 distance_limit에 도달하면 탐색 중지
        pass

    def to_csv(self, dir_path: str) -> None:
        pass


class AdjacencyMatrix:
    """adj_mx.pkl"""

    def __init__(
        self,
        distances: pd.DataFrame,
        sensor_ids: List[str],
    ) -> None:
        self.distance_df = distances
        self.sensor_ids = sensor_ids

    def to_pickle(self, dir_path: str) -> None:
        pass

    def get_adjacency_matrix(
        distance_df: pd.DataFrame, sensor_ids: List[str], normalized_k=0.1
    ):
        """
        :param normalized_k: entries that become lower than normalized_k after normalization are set to zero for sparsity.
        :return: adjacency matrix
        """
        num_sensors = len(sensor_ids)
        dist_mx = np.zeros((num_sensors, num_sensors), dtype=np.float32)
        dist_mx[:] = np.inf
        # Builds sensor id to index map.
        sensor_id_to_ind = {}
        for i, sensor_id in enumerate(sensor_ids):
            sensor_id_to_ind[sensor_id] = i
        # Fills cells in the matrix with distances.
        for row in distance_df.values:
            if row[0] not in sensor_id_to_ind or row[1] not in sensor_id_to_ind:
                continue
            dist_mx[sensor_id_to_ind[row[0]], sensor_id_to_ind[row[1]]] = row[2]

        # Calculates the standard deviation as theta.
        distances = dist_mx[~np.isinf(dist_mx)].flatten()
        std = distances.std()
        adj_mx = np.exp(-np.square(dist_mx / std))
        # Make the adjacent matrix symmetric by taking the max.
        # adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])

        # Sets entries that lower than a threshold, i.e., k, to zero for sparsity.
        adj_mx[adj_mx < normalized_k] = 0

        return adj_mx
