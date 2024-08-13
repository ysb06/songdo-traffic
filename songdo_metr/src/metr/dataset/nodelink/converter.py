import logging
import os
from functools import reduce
from typing import List

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

INCHEON_CODE = ["161", "162", "163", "164", "165", "166", "167", "168", "169"]

logger = logging.getLogger(__name__)


class NodeLinkData:
    def __init__(
        self,
        node_data: gpd.GeoDataFrame,
        link_data: gpd.GeoDataFrame,
        turn_data: pd.DataFrame,
    ) -> None:
        self.node_data = node_data
        self.link_data = link_data
        self.turn_data = turn_data

    def filter_by_gu_codes(self, gu_code_list: List[str]):
        filtered_node_data = self.__filter_data(
            self.node_data, ["NODE_ID"], gu_code_list
        )
        filtered_link_data = self.__filter_data(
            self.link_data, ["LINK_ID", "F_NODE", "T_NODE"], gu_code_list
        )
        filtered_turn_data = self.__filter_data(
            self.turn_data, ["NODE_ID", "ST_LINK", "ED_LINK"], gu_code_list
        )

        return NodeLinkData(filtered_node_data, filtered_link_data, filtered_turn_data)

    def export(self, output_dir: str):
        logger.info("Exporting datasets...")
        self.node_data.to_file(
            os.path.join(output_dir, "imc_node.shp"), encoding="utf-8"
        )
        self.link_data.to_file(
            os.path.join(output_dir, "imc_link.shp"), encoding="utf-8"
        )
        turn_data = gpd.GeoDataFrame(self.turn_data)
        turn_data.to_file(
            os.path.join(output_dir, "imc_turninfo.dbf"), encoding="utf-8"
        )
        logger.info("Exporting completed.")

    def __filter_data(
        self, data: gpd.GeoDataFrame, columns: List[str], code_list: List[str]
    ) -> gpd.GeoDataFrame:
        condition_list = [data[column].str[:3].isin(code_list) for column in columns]
        return data[reduce(lambda x, y: x | y, condition_list)]


class NodeLink(NodeLinkData):
    def __init__(self, root_path, encoding="cp949") -> None:
        self.node_path = os.path.join(root_path, "MOCT_NODE.shp")
        self.link_path = os.path.join(root_path, "MOCT_LINK.shp")
        self.turn_path = os.path.join(root_path, "TURNINFO.dbf")

        logger.info("Loading node data...")
        node_data: gpd.GeoDataFrame = gpd.read_file(self.node_path, encoding=encoding)
        logger.info(f"Done: {type(node_data)}")
        logger.info("Loading link data...")
        link_data: gpd.GeoDataFrame = gpd.read_file(self.link_path, encoding=encoding)
        logger.info(f"Done: {type(link_data)}")
        logger.info("Loading turning data...")
        turn_data: pd.DataFrame = gpd.read_file(self.turn_path, encoding=encoding)
        logger.info(f"Done: {type(turn_data)}")

        super().__init__(node_data, link_data, turn_data)


def get_sensor_node_list(
    traffic_data: pd.DataFrame,
    sensor_attr: List[str] = [
        "road_bhf_fclts_id",
        "road_bhf_fclts_nm",
        "instl_lc_nm",
        "road_bhf_area_nm",
    ],
    sensor_coord_attr: List[str] = ["lo_ycrd", "la_xcrd"],
) -> gpd.GeoDataFrame:
    unique_sensors = traffic_data[sensor_attr + sensor_coord_attr].drop_duplicates()
    seonsor_geometry = [
        Point(xy)
        for xy in zip(
            unique_sensors[sensor_coord_attr[0]], unique_sensors[sensor_coord_attr[1]]
        )
    ]
    sensor_gdf = gpd.GeoDataFrame(
        unique_sensors[sensor_attr], geometry=seonsor_geometry
    )
    sensor_gdf.crs = "EPSG:4326"

    return sensor_gdf
