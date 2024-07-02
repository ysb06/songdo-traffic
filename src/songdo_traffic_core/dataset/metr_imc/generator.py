import logging
import os
from typing import Any, List, Optional

import geopandas as gpd
import pandas as pd

from .converter import (
    AdjacencyMatrix,
    GraphSensorLocations,
    MetrIds,
    MetrImc,
    DistancesImc,
)

logger = logging.getLogger(__name__)


class MetrImcDatasetGenerator:
    def __init__(self, nodelink_dir: str, imcrts_dir: str) -> None:
        self.intersection_gdf = gpd.read_file(
            os.path.join(nodelink_dir, "imc_node.shp")
        )
        self.road_gdf = gpd.read_file(os.path.join(nodelink_dir, "imc_link.shp"))
        self.turninfo_gdf = gpd.read_file(
            os.path.join(nodelink_dir, "imc_turninfo.shp")
        )

        self.traffic_df = pd.read_pickle(os.path.join(imcrts_dir, "imcrts_data.pkl"))

    def generate(self, output_dir: str):
        os.makedirs(os.path.join(output_dir, "miscellaneous"), exist_ok=True)

        logger.info("metr-imc.h5")
        metr_imc = MetrImc(self.traffic_df, self.road_gdf)
        metr_imc.to_hdf(output_dir)
        metr_imc.to_excel(f"{output_dir}/miscellaneous")

        # id는 교통량 데이터에 있는 것만!
        logger.info("metr_ids.txt")
        sensor_ids = MetrIds(metr_imc.road_ids)
        sensor_ids.to_txt(output_dir)

        logger.info("graph_sensor_locations.csv")
        sensor_loc = GraphSensorLocations(self.road_gdf, metr_imc.road_ids)
        sensor_loc.to_csv(output_dir)

        logger.info("distances_imc_2023.csv")
        distances_imc = DistancesImc(
            self.road_gdf, self.turninfo_gdf, metr_imc.road_ids
        )
        distances_imc.to_csv(output_dir)

        logger.info("adj_mx.pkl")
        adj_mx = AdjacencyMatrix(distances_imc.distances, metr_imc.road_ids)
        adj_mx.to_pickle(output_dir)

        # Todo: adjacency matrix
        # Todo: 모델 학습 돌려보기
        # Todo: W_metrimc, SE_metrimc는 정확히 무엇인지 파악하고 작업. 일단은 우선 순위 낮음.


class MetrImcSubsetGenerator:
    """Class for generating a subset of the Metr-IMC dataset."""

    def __init__(
        self,
        nodelink_dir: str = "./datasets/metr-imc/nodelink",
        road_gdf_filename: str = "imc_link.shp",
        turninfo_gdf_filename: str = "imc_turninfo.shp",
        imcrts_dir: str = "./datasets/metr-imc/imcrts",
        imcrts_filename: str = "imcrts_data.pkl",
        metr_imc_dir: str = "./datasets/metr-imc",
        metr_imc_filename: str = "metr-imc.h5",
        metr_ids_filename: str = "metr_ids.txt",
        graph_sensor_loc_filename: str = "graph_sensor_locations.csv",
        distances_imc_filename: str = "distances_imc_2023.csv",
        adj_mx_filename: str = "adj_mx.pkl",
    ) -> None:
        self.nodelink_dir = nodelink_dir
        self.imcrts_dir = imcrts_dir
        self.metr_imc_dir = metr_imc_dir

        self.roads_gdf = gpd.read_file(os.path.join(nodelink_dir, road_gdf_filename))
        self.turninfo_gdf = gpd.read_file(
            os.path.join(nodelink_dir, turninfo_gdf_filename)
        )
        self.imcrts_df = pd.read_pickle(os.path.join(imcrts_dir, imcrts_filename))
        self.metr_imc_path = os.path.join(metr_imc_dir, metr_imc_filename)
        self.metr_imc_df: Optional[pd.DataFrame] = None
        if os.path.exists(self.metr_imc_path):
            self.metr_imc_df = pd.read_hdf(self.metr_imc_path)
        self.sensor_ids_path = os.path.join(metr_imc_dir, metr_ids_filename)
        self.metr_id_list: Optional[List[str]] = None
        if os.path.exists(self.sensor_ids_path):
            with open(self.sensor_ids_path, "r") as f:
                self.metr_id_list = f.read().split(",")
        self.graph_sensor_loc_path = os.path.join(
            metr_imc_dir, graph_sensor_loc_filename
        )
        self.graph_sensor_loc: Optional[pd.DataFrame] = None
        if os.path.exists(self.graph_sensor_loc_path):
            self.graph_sensor_loc = pd.read_csv(self.graph_sensor_loc_path)
        self.distances_imc_path = os.path.join(metr_imc_dir, distances_imc_filename)
        self.distances_imc: Optional[pd.DataFrame] = None
        if os.path.exists(self.distances_imc_path):
            self.distances_imc = pd.read_csv(self.distances_imc_path)
        self.adj_mx_path = os.path.join(metr_imc_dir, adj_mx_filename)
        self.adj_mx: Optional[Any] = None
        if os.path.exists(self.adj_mx_path):
            self.adj_mx = pd.read_pickle(self.adj_mx_path)

    def generate_subset(
        self, targets: List[str], output_dir: str = "./", need_interpolate: bool = True
    ) -> None:
        self.__generate_all()

        logger.info(f"Start generating subset...")
        os.makedirs(output_dir, exist_ok=True)

        # 새 데이터 생성
        # METR-IMC
        metr_imc_path = os.path.join(output_dir, os.path.split(self.metr_imc_path)[1])
        logger.info(f"Generating {metr_imc_path}...")
        metr_imc = self.metr_imc_df[targets]
        metr_imc.to_hdf(metr_imc_path, key="data")

        # Sensor IDs
        logger.info(
            f"Generating {os.path.join(output_dir, os.path.split(self.sensor_ids_path)[1])}..."
        )
        sensor_ids = MetrIds(targets)
        sensor_ids.to_txt(output_dir, os.path.split(self.sensor_ids_path)[1])

        # Graph Sensor Locations
        logger.info(
            f"Generating {os.path.join(output_dir, os.path.split(self.graph_sensor_loc_path)[1])}..."
        )
        sensor_loc = GraphSensorLocations(self.roads_gdf, targets)
        sensor_loc.to_csv(output_dir, os.path.split(self.graph_sensor_loc_path)[1])

        # Distances IMC
        logger.info(
            f"Generating {os.path.join(output_dir, os.path.split(self.distances_imc_path)[1])}..."
        )
        distances_imc = DistancesImc(
            self.roads_gdf,
            self.turninfo_gdf,
            targets,
        )
        distances_imc.to_csv(output_dir, os.path.split(self.distances_imc_path)[1])

        # Adjacency Matrix
        logger.info(
            f"Generating {os.path.join(output_dir, os.path.split(self.adj_mx_path)[1])}..."
        )
        adj_mx = AdjacencyMatrix(self.distances_imc, targets)
        adj_mx.to_pickle(output_dir, os.path.split(self.adj_mx_path)[1])

    def __generate_all(self):
        if self.metr_imc_df is None:
            logger.info("Generating metr-imc.h5...")
            metr_imc = MetrImc(self.imcrts_df, self.roads_gdf)
            self.metr_imc_df = metr_imc.data
            metr_imc.to_hdf(*os.path.split(self.metr_imc_path))
        else:
            logger.info("metr-imc.h5 already exists")

        road_ids = self.metr_imc_df.columns.tolist()

        if self.metr_id_list is None:
            logger.info("Generating metr_ids.txt...")
            sensor_ids = MetrIds(road_ids)
            self.metr_id_list = sensor_ids.id_list
            sensor_ids.to_txt(*os.path.split(self.sensor_ids_path))
        else:
            logger.info("metr_ids.txt already exists")

        if self.graph_sensor_loc is None:
            logger.info("Generating graph_sensor_locations.csv...")
            sensor_loc = GraphSensorLocations(self.roads_gdf, road_ids)
            self.graph_sensor_loc = sensor_loc.result
            sensor_loc.to_csv(*os.path.split(self.graph_sensor_loc_path))
        else:
            logger.info("graph_sensor_locations.csv already exists")

        if self.distances_imc is None:
            logger.info("Generating distances_imc_2024.csv...")
            distances_imc = DistancesImc(
                self.roads_gdf,
                self.turninfo_gdf,
                road_ids,
            )
            self.distances_imc = distances_imc.distances
            distances_imc.to_csv(*os.path.split(self.distances_imc_path))
        else:
            logger.info("distances_imc_2024.csv already exists")

        if self.adj_mx is None:
            adj_mx = AdjacencyMatrix(self.distances_imc, road_ids)
            self.adj_mx = adj_mx.adj_mx
            adj_mx.to_pickle(*os.path.split(self.adj_mx_path))
        else:
            logger.info("adj_mx.pkl already exists")
