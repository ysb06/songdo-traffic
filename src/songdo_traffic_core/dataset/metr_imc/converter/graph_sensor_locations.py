import sys
import argparse
import logging
import os
from typing import Dict, List, Optional, Set

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point
from tqdm import tqdm

logger = logging.getLogger(__name__)

GRAPH_SENSOR_LOCATIONS_DIR = "./datasets/metr-imc/"
GRAPH_SENSOR_LOCATIONS_FILE_NAME = "graph_sensor_locations.csv"
GRAPH_SENSOR_LOCATIONS_SHAPEFILE_DIR = (
    "./datasets/metr-imc/miscellaneous/graph_sensor_location/"
)
GRAPH_SENSOR_LOCATIONS_SHAPEFILE_FILE_NAME = "sensor_node.shp"


def calculate_mean_coords(geometry):
    """GeoPandas의 geometry에서 Line을 나타내는 두 점의 중앙 값을 추출

    Args:
        geometry (_type_): 도로를 나타내는 링크 Line 리스트

    Returns:
        pandas.Series: 두 점의 EPSG:4326 기반 위치 목록
    """
    x_coords = [point[0] for point in geometry.coords]
    y_coords = [point[1] for point in geometry.coords]
    mean_x = np.mean(x_coords)
    mean_y = np.mean(y_coords)
    return pd.Series([mean_x, mean_y], index=["longitude", "latitude"])


class GraphSensorLocations:
    """
    표준노드링크의 Link Shapefile로부터 센서의 실제 위치인 graph_sensor_locations.csv 파일을 생성
    """

    def __init__(
        self, road_data: gpd.GeoDataFrame, target_ids: Optional[List[str]] = None
    ) -> None:
        road_data = road_data.to_crs(epsg=4326)

        self.result = pd.DataFrame(columns=["sensor_id", "latitude", "longitude"])
        self.result["sensor_id"] = road_data["LINK_ID"]
        self.result[["longitude", "latitude"]] = road_data["geometry"].apply(
            calculate_mean_coords
        )
        if target_ids is not None:
            self.result = self.result[self.result["sensor_id"].isin(target_ids)]

    def to_csv(
        self, dir_path: str, filename: str = GRAPH_SENSOR_LOCATIONS_FILE_NAME
    ) -> None:
        path = os.path.join(dir_path, filename)
        logger.info(f"Saving sensor locations to {path}...")
        self.result.to_csv(path, index_label="index")
        logger.info("Complete")


class SensorView:
    """
    생성된 graph_sensor_locations.csv의 결과를 Shapefile로 다시 저장
    """

    def __init__(self, graph_sensor_locations_path: str) -> None:
        self.gdf_raw = self.__get_node_gpd(graph_sensor_locations_path)
        self.column_filter: Optional[List[str]] = None

    def __get_node_gpd(self, graph_sensor_locations_path: str) -> gpd.GeoDataFrame:
        gsl_df = pd.read_csv(
            graph_sensor_locations_path,
            index_col="index",
            dtype={"sensor_id": str, "longitude": float, "latitude": float},
        )
        return gpd.GeoDataFrame(
            gsl_df["sensor_id"],
            geometry=gpd.points_from_xy(gsl_df["longitude"], gsl_df["latitude"]),
            crs="EPSG:4326",
        )

    def set_filter(self, columns: List[str]) -> None:
        self.column_filter = columns

    def export_to_file(
        self, folder_path: str, file_name: str = "sensor_nodes.shp"
    ) -> None:
        if self.column_filter is not None:
            result = self.gdf_raw[self.gdf_raw["sensor_id"].isin(self.column_filter)]
        else:
            result = self.gdf_raw
        result.to_file(os.path.join(folder_path, file_name), encoding="utf-8")


class SensorNetworkView:
    def __init__(self, distance_csv_path: str, sensor_loc_csv_path: str) -> None:
        self.sensor_distance_df = pd.read_csv(distance_csv_path)
        self.sensor_location_df = pd.read_csv(sensor_loc_csv_path, index_col="index")

        self.node_data = self.__generate_node_data(self.sensor_location_df)

        node_dict = self.node_data.set_index("NODE_ID")["geometry"].to_dict()
        self.link_data = self.__generate_link_data(
            node_dict, set(self.node_data["NODE_ID"])
        )

    def to_file(
        self,
        output_dir: str,
        sensor_node_filename: str = "sensor_nodes.shp",
        sensor_link_file_name: str = "sensor_links.shp",
    ) -> None:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        sensor_node_path = os.path.join(output_dir, sensor_node_filename)
        sensor_link_path = os.path.join(output_dir, sensor_link_file_name)
        self.node_data.to_file(sensor_node_path, encoding="utf-8")
        self.link_data.to_file(sensor_link_path, encoding="utf-8")

    def __generate_node_data(self, sensor_loc_df: pd.DataFrame) -> gpd.GeoDataFrame:
        data = gpd.GeoDataFrame(
            sensor_loc_df["sensor_id"],
            geometry=gpd.points_from_xy(
                sensor_loc_df["longitude"], sensor_loc_df["latitude"]
            ),
            crs="EPSG:4326",
        )
        data = data.rename(columns={"sensor_id": "NODE_ID"})

        return data

    def __generate_link_data(
        self,
        node_dict: Dict[str, Point],
        node_set: Set[str],
        link_list: gpd.GeoDataFrame = None,
        delete_zero_length: bool = True,
    ) -> gpd.GeoDataFrame:
        # 굳이 매개변수 받지 말고 self 매개변수로 받아서 처리하는게 더 좋을 것 같음
        link_data = self.sensor_distance_df[
            (self.sensor_distance_df["from"].isin(node_set))
            & (self.sensor_distance_df["to"].isin(node_set))
        ]
        link_data.reset_index(drop=True, inplace=True)

        if link_list is None:
            pass  # 현재는 미구현, link_list에서 LINK_ID를 검색하여 self.link_data.index를 생성하는 방식으로 구현
            # 표준노드링크를 사용한 데이터에서 사용 예정

        data = {"LINK_ID": [], "F_NODE": [], "T_NODE": [], "COST": [], "geometry": []}
        for idx, from_node, to_node, cost in tqdm(
            link_data.itertuples(),
            total=link_data.shape[0],
        ):
            link_line = LineString([node_dict[from_node], node_dict[to_node]])
            if delete_zero_length and link_line.length <= 0:
                continue

            data["LINK_ID"].append(idx)
            data["F_NODE"].append(from_node)
            data["T_NODE"].append(to_node)
            data["COST"].append(cost)  # This may same as geometry length
            data["geometry"].append(link_line)

        data = gpd.GeoDataFrame(data, crs="EPSG:4326")

        return data


# METR-LA의 속도는 mile/h라고 하지만, Distance의 경우 m 단위로 보임
# 아마도 Distance의 경우 처음 데이터 이후에 추가된 것으로 보임


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Converter for graph_sensor_location")
    parser.add_argument(
        "--sensor_file",
        type=str,
        default="./datasets/metr-imc-296-interpolation-full/graph_sensor_locations.csv",
        help="File path for sensor locations",
    )
    parser.add_argument(
        "--distance_file",
        type=str,
        default="./datasets/metr-imc-296-interpolation-full/distances_imc_2023.csv",
        help="File path for sensor locations",
    )
    parser.add_argument(
        "--sensor_network_dir",
        nargs="?",
        const="./datasets/metr-imc/miscellaneous/metr-imc-296-int-full/",
        help="File path for Shapefile",
    )
    args = parser.parse_args()

    if args.sensor_network_dir is not None:
        SensorNetworkView(args.distance_file, args.sensor_file).to_file(args.sensor_network_dir)
    else:
        logger.warning("Nothing to do")
