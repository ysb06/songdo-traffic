from datetime import datetime
from typing import Callable, List, Optional, Tuple
import pandas as pd
import os
import logging

import geopandas as gpd

from metr.dataset.metr_imc.converter.adj_mx import AdjacencyMatrixData
from metr.dataset.metr_imc.converter.distance_imc import DistancesData
from metr.dataset.metr_imc.converter.metr_ids import MetrIds
from .converter.metr_imc import MetrImcTrafficData

logger = logging.getLogger(__name__)


class MetrBase:
    def __init__(
        self,
        traffic_data: MetrImcTrafficData,
        road_ids: MetrIds,
        distances: DistancesData,
        adj_mx: AdjacencyMatrixData,
    ) -> None:
        self.traffic_data = traffic_data
        self.road_ids = road_ids
        self.distances = distances
        self.adj_mx = adj_mx


class DataPath:
    def __init__(self, directory: str, filename: str) -> None:
        self.directory = directory
        self.filename = filename

    @property
    def exists(self) -> bool:
        return os.path.exists(self)
    
    @property
    def tuple(self) -> Tuple[str, str]:
        return self.directory, self.filename

    def __str__(self) -> str:
        return os.path.join(self.directory, self.filename)

    def __repr__(self) -> str:
        return os.path.join(self.directory, self.filename)
    
    def __fspath__(self) -> str:
        return os.path.join(self.directory, self.filename)


class Metr:
    def __init__(
        self,
        target_root_dir: str,
        traffic_data_filename: str = "metr-imc.h5",
        road_ids_filename: str = "metr_ids.txt",
        distances_filename: str = "distances_imc_2023.csv",
        adj_mx_filename: str = "adj_mx.pkl",
    ) -> None:
        self.tdat_path: DataPath = DataPath(target_root_dir, traffic_data_filename)
        self.rdid_path: DataPath = DataPath(target_root_dir, road_ids_filename)
        self.dist_path: DataPath = DataPath(target_root_dir, distances_filename)
        self.admx_path: DataPath = DataPath(target_root_dir, adj_mx_filename)

        print("OK")
        data = None
        if self.tdat_path.exists:
            data = pd.read_hdf(self.tdat_path)
        traffic_data = MetrImcTrafficData(data)

        data = None
        if self.rdid_path.exists:
            with open(self.rdid_path, "r") as f:
                data = f.read().split(",")
        road_ids = MetrIds(data)

        data = None
        if self.dist_path.exists:
            data = pd.read_csv(self.dist_path)
        distances = DistancesData(data)

        adj_mx = AdjacencyMatrixData()
        if self.admx_path.exists:
            adj_mx.read_pkl(self.admx_path)

        self.raw_base = MetrBase(traffic_data, road_ids, distances, adj_mx)

    @property
    def is_complete(self) -> bool:
        return (
            self.raw_base.traffic_data.data_exists
            and self.raw_base.road_ids.data_exists
            and self.raw_base.distances.data_exists
            and self.raw_base.adj_mx.data_exists
        )

    def generate_all_data(
        self,
        imcrts_path: Optional[str],
        road_data_path: Optional[str],
        turn_info_path: Optional[str],
        target_columns: Optional[List[str]] = None,
        target_periods: Optional[List[datetime]] = None,
        distance_limit: float = 9000,
    ) -> None:
        logger.info("Generating traffic data from IMCRTS...")
        self.raw_base.traffic_data.import_from_imcrts(imcrts_path)

        if target_columns is not None:
            self.raw_base.traffic_data.select_columns(target_columns)
        if target_periods is not None:
            self.raw_base.traffic_data.select_period(idx_list=target_periods)

        logger.info("Generating road ids from the traffic data...")
        self.raw_base.road_ids.import_from_traffic_data(self.raw_base.traffic_data.data)

        logger.info("Generating distances data...")
        road_data = gpd.read_file(road_data_path)
        turn_info = gpd.read_file(turn_info_path)
        self.raw_base.distances.generate_graph(
            road_data,
            turn_info,
            self.raw_base.road_ids.id_list,
            distance_limit=distance_limit,
        )

        logger.info("Generating adjacency matrix...")
        self.raw_base.adj_mx.generate_adj_mx(
            self.raw_base.distances.data,
            self.raw_base.road_ids.id_list,
        )

        logger.info("Data generation complete.")
        logger.info(f"Traffic Data --> {self.raw_base.traffic_data.data.shape}")
        logger.info(f"Road IDs --> {len(self.raw_base.road_ids.id_list)}")
        logger.info(f"Distances --> {self.raw_base.distances.data.shape}")
        logger.info(f"Adjacency Matrix --> {self.raw_base.adj_mx.data[2].shape}")

    def export_data(self, overwrite: bool = False, skip_exist: bool = True) -> None:
        def save_file(data_path: DataPath, save_method: Callable[[str, str], None]) -> None:
            if overwrite or not os.path.exists(data_path):
                save_method(*data_path.tuple)
            elif not skip_exist:
                raise Exception(f"{data_path} already exists. Set overwrite to True to overwrite it.")
            else:
                logger.info(f"{data_path} already exists. Skipping...")

        files_methods = {
            self.tdat_path: self.raw_base.traffic_data.to_hdf,
            self.rdid_path: self.raw_base.road_ids.to_txt,
            self.dist_path: self.raw_base.distances.to_csv,
            self.admx_path: self.raw_base.adj_mx.to_pickle,
        }

        if overwrite:
            logger.info("Force generating files...")

        for data_path, save_method in files_methods.items():
            save_file(data_path, save_method)

        


class MetrDataset:
    def __init__(self) -> None:
        pass
