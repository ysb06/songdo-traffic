import logging
import os

import geopandas as gpd

from metr.components import (
    AdjacencyMatrix,
    DistancesImc,
    IdList,
    Metadata,
    SensorLocations,
    TrafficData,
)
from metr.imcrts.collector import IMCRTSCollector
from metr.nodelink.converter import NodeLink
from metr.nodelink.downloader import download_nodelink
from metr.utils import PathConfig


logger = logging.getLogger(__name__)
PATH_CONF = PathConfig.from_yaml("../config.yaml")
PATH_CONF.create_directories()


# Other Settings
PDP_KEY = os.environ.get("PDP_KEY")
NODELINK_RAW_URL = (
    "https://www.its.go.kr/opendata/nodelinkFileSDownload/DF_195/0"  # 2024-03-25
)
TARGET_REGION_CODES = [
    "161",
    "162",
    "163",
    "164",
    "165",
    "166",
    "167",
    "168",
    "169",
]  # All Incheon Regions
IMCRTS_START_DATE = "20221101"
IMCRTS_END_DATE = "20250310"


def generate_subset_dataset(
    target_nodelink: list[str],
    save_dir_path: str,
):
    metr_imc_filename = PATH_CONF.raw["dataset"]["filenames"]["metr_imc"]
    sensor_ids_filename = PATH_CONF.raw["dataset"]["filenames"]["sensor_ids"]
    metadata_filename = PATH_CONF.raw["dataset"]["filenames"]["metadata"]
    sensor_locations_filename = PATH_CONF.raw["dataset"]["filenames"][
        "sensor_locations"
    ]
    distances_filename = PATH_CONF.raw["dataset"]["filenames"]["distances"]
    adjacency_matrix_filename = PATH_CONF.raw["dataset"]["filenames"][
        "adjacency_matrix"
    ]

    metr_imc_save_path = os.path.join(save_dir_path, metr_imc_filename)
    sensor_ids_save_path = os.path.join(save_dir_path, sensor_ids_filename)
    metadata_save_path = os.path.join(save_dir_path, metadata_filename)
    sensor_locations_save_path = os.path.join(save_dir_path, sensor_locations_filename)
    distances_save_path = os.path.join(save_dir_path, distances_filename)
    adj_mx_save_path = os.path.join(save_dir_path, adjacency_matrix_filename)

    traffic_data = TrafficData.import_from_hdf(PATH_CONF.metr_imc_path)
    traffic_data.select_sensors(target_nodelink)
    traffic_data.to_hdf(metr_imc_save_path)

    generate_dataset(
        traffic_data_path=metr_imc_save_path,
        ids_output_path=sensor_ids_save_path,
        metadata_output_path=metadata_save_path,
        sensor_locations_output_path=sensor_locations_save_path,
        distances_output_path=distances_save_path,
        adj_mx_output_path=adj_mx_save_path,
    )


def generate_raw_dataset():
    # Generating Core Files
    generate_nodelink_raw()
    generate_imcrts_raw()
    generate_metr_imc_raw()
    generate_dataset()

    # Generating Misc
    generate_metr_imc_shapefile()
    generate_distances_shapefile()


def generate_distances_shapefile(
    distances_path: str = PATH_CONF.distances_path,
    sensor_locations_path: str = PATH_CONF.sensor_locations_path,
    output_path: str = PATH_CONF.distances_shapefile_path,
):
    distances = DistancesImc.import_from_csv(distances_path)
    sensor_locations = SensorLocations.import_from_csv(sensor_locations_path)
    distances.to_shapefile(sensor_locations.data, filepath=output_path)


def generate_dataset(
    traffic_data_path: str = PATH_CONF.metr_imc_path,
    nodelink_link_path: str = PATH_CONF.nodelink_link_path,
    nodelink_turn_path: str = PATH_CONF.nodelink_turn_path,
    ids_output_path: str = PATH_CONF.sensor_ids_path,
    metadata_output_path: str = PATH_CONF.metadata_path,
    sensor_locations_output_path: str = PATH_CONF.sensor_locations_path,
    distances_output_path: str = PATH_CONF.distances_path,
    adj_mx_output_path: str = PATH_CONF.adj_mx_path,
):
    traffic_data = TrafficData.import_from_hdf(traffic_data_path)

    # Sensor IDs
    metr_ids = IdList(traffic_data.data.columns.to_list())
    metr_ids.to_txt(ids_output_path)

    # Metadata
    metadata = Metadata.import_from_nodelink(nodelink_link_path)
    metadata.sensor_filter = metr_ids.data  # 수정 필요
    metadata.to_hdf(metadata_output_path)

    # Sensor Locations
    sensor_locations = SensorLocations.import_from_nodelink(nodelink_link_path)
    sensor_locations.sensor_filter = metr_ids.data
    sensor_locations.to_csv(sensor_locations_output_path)

    # Distances
    distances = DistancesImc.import_from_nodelink(
        nodelink_link_path,
        nodelink_turn_path,
        target_ids=metr_ids.data,
        distance_limit=9000,
    )
    distances.to_csv(distances_output_path)

    # Adjacency Matrix
    adj_mx: AdjacencyMatrix = AdjacencyMatrix.import_from_components(
        metr_ids, distances
    )
    adj_mx.to_pickle(adj_mx_output_path)


def generate_nodelink_raw(
    nodelink_url: str = NODELINK_RAW_URL,
    region_codes: list[str] = TARGET_REGION_CODES,
    download_target_dir: str = PATH_CONF.nodelink_dir_path,
    node_output_path: str = PATH_CONF.nodelink_node_path,
    link_output_path: str = PATH_CONF.nodelink_link_path,
    turn_output_path: str = PATH_CONF.nodelink_turn_path,
):
    logger.info("Downloading Node-Link Data...")
    nodelink_raw_path = download_nodelink(download_target_dir, nodelink_url)
    nodelink_data = NodeLink(nodelink_raw_path).filter_by_gu_codes(region_codes)
    nodelink_data.export(
        node_output_path=node_output_path,
        link_output_path=link_output_path,
        turn_output_path=turn_output_path,
    )
    logger.info("Downloading Done")


def generate_imcrts_raw(
    api_key: str = PDP_KEY,
    start_date: str = IMCRTS_START_DATE,
    end_date: str = IMCRTS_END_DATE,
    imcrts_output_path: str = PATH_CONF.imcrts_path,
):
    logger.info("Collecting IMCRTS Data...")
    collector = IMCRTSCollector(
        api_key=api_key,
        start_date=start_date,
        end_date=end_date,
    )
    collector.collect(ignore_empty=True)
    collector.to_pickle(imcrts_output_path)
    logger.info("Collecting Done")


def generate_metr_imc_raw(
    road_data_path: str = PATH_CONF.nodelink_link_path,
    traffic_data_path: str = PATH_CONF.imcrts_path,
    metr_imc_path: str = PATH_CONF.metr_imc_path,
):
    road_data: gpd.GeoDataFrame = gpd.read_file(road_data_path)
    traffic_data = TrafficData.import_from_pickle(traffic_data_path)

    logger.info("Matching Link IDs...")
    traffic_data.select_sensors(road_data["LINK_ID"].tolist())
    traffic_data.to_hdf(metr_imc_path)
    logger.info("Matching Done")


def generate_metr_imc_shapefile(
    metr_imc_path: str = PATH_CONF.metr_imc_path,
    node_link_path: str = PATH_CONF.nodelink_link_path,
    output_path: str = PATH_CONF.metr_shapefile_path,
):
    traffic_data = TrafficData.import_from_hdf(metr_imc_path)
    road_data: gpd.GeoDataFrame = gpd.read_file(node_link_path)
    traffic_link_ids = set(traffic_data.data.columns)
    filtered_roads = road_data[road_data["LINK_ID"].isin(traffic_link_ids)].copy()
    filtered_roads.to_file(output_path)
