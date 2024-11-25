import logging
import os

import geopandas as gpd

from metr.components import (
    TrafficData,
    Metadata,
    SensorLocations,
    DistancesImc,
    AdjacencyMatrix,
    IdList,
)
from metr.processors import *
from metr.dataset import generate_file_set

logger = logging.getLogger(__name__)

SUBSET_TARGET_SENSOR_PATH = os.path.join(
    SUBSET_MISCELLANEOUS_DIR, SUBSET_TARGET_SENSOR_FILENAME
)
ORIGIN_TRAFFIC_PATH = os.path.join(RAW_DATASET_ROOT_DIR, TRAFFIC_FILENAME)
ORIGIN_TRAFFIC_TRAINING_PATH = os.path.join(RAW_DATASET_ROOT_DIR, TRAFFIC_TRAINING_FILENAME)
ORIGIN_TRAFFIC_TEST_PATH = os.path.join(RAW_DATASET_ROOT_DIR, TRAFFIC_TEST_FILENAME)

SUBSET_TRAFFIC_PATH = os.path.join(SUBSET_DATASET_ROOT_DIR, TRAFFIC_FILENAME)
SUBSET_TRAFFIC_TRAINING_PATH = os.path.join(SUBSET_DATASET_ROOT_DIR, TRAFFIC_TRAINING_FILENAME)
SUBSET_TRAFFIC_TEST_PATH = os.path.join(SUBSET_DATASET_ROOT_DIR, TRAFFIC_TEST_FILENAME)

SUBSET_SENSOR_IDS_PATH = os.path.join(SUBSET_DATASET_ROOT_DIR, SENSOR_IDS_FILENAME)
SUBSET_METADATA_PATH = os.path.join(SUBSET_DATASET_ROOT_DIR, METADATA_FILENAME)
SUBSET_SENSOR_LOCATIONS_PATH = os.path.join(SUBSET_DATASET_ROOT_DIR, SENSOR_LOCATIONS_FILENAME)
SUBSET_DISTANCES_PATH = os.path.join(SUBSET_DATASET_ROOT_DIR, DISTANCES_FILENAME)
SUBSET_ADJ_MX_PATH = os.path.join(SUBSET_DATASET_ROOT_DIR, ADJ_MX_FILENAME)

def run_process():
    os.makedirs(SUBSET_MISCELLANEOUS_DIR, exist_ok=True)

    filter_traffic_data()
    generate_subset_file_set()


def filter_traffic_data(
    subset_target_sensor_path: str = SUBSET_TARGET_SENSOR_PATH,
    original_traffic_raw_path=ORIGIN_TRAFFIC_PATH,
    original_traffic_training_path=ORIGIN_TRAFFIC_TRAINING_PATH,
    original_traffic_test_path=ORIGIN_TRAFFIC_TEST_PATH,
    subset_traffic_raw_path=SUBSET_TRAFFIC_PATH,
    subset_traffic_training_path=SUBSET_TRAFFIC_TRAINING_PATH,
    subset_traffic_test_path=SUBSET_TRAFFIC_TEST_PATH,
):
    target_sensors: gpd.GeoDataFrame = gpd.read_file(subset_target_sensor_path)

    small_filter = target_sensors["LINK_ID"].astype(str).to_list()
    traffic_raw_data = TrafficData.import_from_hdf(original_traffic_raw_path)
    traffic_training_data = TrafficData.import_from_hdf(original_traffic_training_path)
    traffic_test_data = TrafficData.import_from_hdf(original_traffic_test_path)

    traffic_raw_data.sensor_filter = small_filter
    traffic_training_data.sensor_filter = small_filter
    traffic_test_data.sensor_filter = small_filter

    traffic_raw_data.to_hdf(subset_traffic_raw_path)
    traffic_training_data.to_hdf(subset_traffic_training_path)
    traffic_test_data.to_hdf(subset_traffic_test_path)

def generate_subset_file_set(
    training_path: TrafficData = SUBSET_TRAFFIC_TRAINING_PATH,
    metadata_path: str = SUBSET_METADATA_PATH,
    sensor_locations_path: str = SUBSET_SENSOR_LOCATIONS_PATH,
    distances_path: str = SUBSET_DISTANCES_PATH,
    adj_mx_path: str = SUBSET_ADJ_MX_PATH,
):
    traffic_data = TrafficData.import_from_hdf(training_path)

    generate_file_set(
        traffic_data,
        NODELINK_TARGET_DIR,
        NODELINK_LINK_FILENAME,
        NODELINK_TURN_FILENAME,
        metadata_path,
        sensor_locations_path,
        distances_path,
        adj_mx_path,
    )