import logging

import geopandas as gpd
import pandas as pd

from .converter import AdjacencyMatrix, GraphSensorLocations, MetrIds, MetrImc

logger = logging.getLogger(__name__)


class MetrImcDatasetGenerator:
    def __init__(self, nodelink_dir: str, imcrts_dir: str) -> None:
        self.intersection_gdf = gpd.read_file(f"{nodelink_dir}/imc_node.shp")
        self.road_gdf = gpd.read_file(f"{nodelink_dir}/imc_link.shp")
        self.turninfo_gdf = gpd.read_file(f"{nodelink_dir}/imc_turninfo.shp")

        self.traffic_df = pd.read_pickle(f"{imcrts_dir}/imcrts_data.pkl")

    def generate(self, output_dir: str):
        logger.info("graph_sensor_locations.csv")
        sensor_loc = GraphSensorLocations(self.road_gdf)
        sensor_loc.to_csv(output_dir)

        logger.info("metr-imc.h5")
        metr_imc = MetrImc(self.traffic_df, self.road_gdf)
        metr_imc.to_hdf(output_dir)
        metr_imc.to_excel(f"{output_dir}/miscellaneous")

        # id는 교통량 데이터에 있는 것만!
        logger.info("metr_ids.txt")
        sensor_ids = MetrIds(metr_imc.road_ids)
        sensor_ids.to_txt(output_dir)
