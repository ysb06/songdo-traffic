import logging
import os
from typing import Any, List, Optional, Tuple

import geopandas as gpd
import pandas as pd

from metr.dataset.interpolator import Interpolator

from .converter.metr_imc import MetrImc
from .converter.adj_mx import AdjacencyMatrix
from .converter.distance_imc import DistancesImc
from .converter.graph_sensor_locations import GraphSensorLocations
from .converter.metr_ids import MetrIds

logger = logging.getLogger(__name__)


class MetrImcDatasetGenerator:
    def __init__(self, nodelink_dir: str, imcrts_dir: str) -> None:
        self.intersection_gdf = gpd.read_file(
            os.path.join(nodelink_dir, "imc_node.shp")
        )
        self.road_gdf = gpd.read_file(os.path.join(nodelink_dir, "imc_link.shp"))
        self.turninfo_gdf: pd.DataFrame = gpd.read_file(
            os.path.join(nodelink_dir, "imc_turninfo.dbf")
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
        metr_imc_extra_filename: str = "metr-imc-extra.h5",
    ) -> None:
        self.nodelink_dir = nodelink_dir
        self.imcrts_dir = imcrts_dir
        self.metr_imc_dir = metr_imc_dir

        self.roads_gdf = gpd.read_file(os.path.join(nodelink_dir, road_gdf_filename))
        self.turninfo_gdf = gpd.read_file(
            os.path.join(nodelink_dir, turninfo_gdf_filename)
        )
        self.imcrts_df: pd.DataFrame = pd.read_pickle(
            os.path.join(imcrts_dir, imcrts_filename)
        )

        self.metr_imc_path = os.path.join(metr_imc_dir, metr_imc_filename)
        self.metr_imc_extra_path = os.path.join(metr_imc_dir, metr_imc_extra_filename)
        self.metr_imc_df: Optional[pd.DataFrame] = None
        if os.path.exists(self.metr_imc_path):
            self.metr_imc_df = pd.DataFrame(pd.read_hdf(self.metr_imc_path))
            self.metr_imc_df.sort_index(axis=1, inplace=True)
            dt_range = pd.date_range(
                start=self.metr_imc_df.index.min(),
                end=self.metr_imc_df.index.max(),
                freq="h",
            )
            self.metr_imc_df = self.metr_imc_df.reindex(dt_range)
        else:
            logger.warning(f"{self.metr_imc_path} not found.")

        self.sensor_ids_path = os.path.join(metr_imc_dir, metr_ids_filename)
        self.metr_id_list: Optional[List[str]] = None
        if os.path.exists(self.sensor_ids_path):
            with open(self.sensor_ids_path, "r") as f:
                self.metr_id_list = f.read().split(",")
        else:
            logger.warning(f"{self.sensor_ids_path} not found.")

        self.graph_sensor_loc_path = os.path.join(
            metr_imc_dir, graph_sensor_loc_filename
        )
        self.graph_sensor_loc: Optional[pd.DataFrame] = None
        if os.path.exists(self.graph_sensor_loc_path):
            self.graph_sensor_loc = pd.read_csv(self.graph_sensor_loc_path)
        else:
            logger.warning(f"{self.graph_sensor_loc_path} not found.")

        self.distances_imc_path = os.path.join(metr_imc_dir, distances_imc_filename)
        self.distances_imc: Optional[pd.DataFrame] = None
        if os.path.exists(self.distances_imc_path):
            self.distances_imc = pd.read_csv(
                self.distances_imc_path,
                dtype={"from": "str", "to": "str", "distance": "float"},
            )
        else:
            logger.warning(f"{self.distances_imc_path} not found.")

        self.adj_mx_path = os.path.join(metr_imc_dir, adj_mx_filename)
        self.adj_mx: Optional[Any] = None
        if os.path.exists(self.adj_mx_path):
            self.adj_mx = pd.read_pickle(self.adj_mx_path)
        else:
            logger.warning(f"{self.adj_mx_path} not found.")

    def process_metr_imc(
        self,
        targets: Optional[List[str]] = None,
        interpolate_filter: Optional[Interpolator] = None,
    ) -> Tuple[pd.DataFrame, List[str]]:
        if self.metr_imc_df is None:
            raise FileNotFoundError(f"{self.metr_imc_path} not found.")

        metr_imc: Optional[pd.DataFrame] = None
        if targets is None:
            targets = self.metr_imc_df.columns.tolist()
            metr_imc = self.metr_imc_df
        else:
            metr_imc = self.metr_imc_df[targets]

        if interpolate_filter is not None:
            logger.info(f"Generate missing value marking data...")
            is_missing_values = metr_imc.isna()
            is_missing_values.to_hdf(self.metr_imc_extra_path, key="is_missing")
            excel_path = os.path.splitext(self.metr_imc_extra_path)[0] + ".xlsx"
            is_missing_values.to_excel(excel_path)
            logger.info(f"Interpolating...")
            metr_imc = interpolate_filter.interpolate(metr_imc)
            logger.info(f"Interpolating Finished!")

        return metr_imc, targets

    def generate_subset(
        self,
        targets: Optional[List[str]] = None,
        output_dir: str = "./",
        interpolate_filter: Optional[Interpolator] = None,
    ) -> None:
        logger.info(f"Start generating subset...")
        os.makedirs(output_dir, exist_ok=True)

        # 새 데이터 생성
        # METR-IMC
        metr_imc_path = os.path.join(output_dir, os.path.split(self.metr_imc_path)[1])
        logger.info(
            f"Generating {metr_imc_path} with {self.metr_imc_df.shape} [Length: {len(targets)}]..."
        )
        metr_imc, targets = self.process_metr_imc(targets, interpolate_filter)
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
        if self.distances_imc is not None:
            adj_mx = AdjacencyMatrix(self.distances_imc, targets)
            adj_mx.to_pickle(output_dir, os.path.split(self.adj_mx_path)[1])
        else:
            logger.warning("Distances IMC not found. Generating failed.")
