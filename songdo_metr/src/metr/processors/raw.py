import logging
import os

import geopandas as gpd

from metr.components import TrafficData
from metr.dataset import generate_file_set
from metr.imcrts.collector import IMCRTSCollector
from metr.nodelink.converter import NodeLink
from metr.nodelink.downloader import download_nodelink
from metr.processors import *

logger = logging.getLogger(__name__)


PDP_KEY = os.environ.get("PDP_KEY")

ROAD_DATA_PATH = os.path.join(NODELINK_TARGET_DIR, NODELINK_LINK_FILENAME)
IMCRTS_PATH = os.path.join(IMCRTS_DIR, IMCRTS_FILENAME)

METR_IMC_PATH = os.path.join(RAW_DATASET_ROOT_DIR, TRAFFIC_FILENAME)
METR_IMC_TRAINING_PATH = os.path.join(RAW_DATASET_ROOT_DIR, TRAFFIC_TRAINING_FILENAME)
METR_IMC_TEST_PATH = os.path.join(RAW_DATASET_ROOT_DIR, TRAFFIC_TEST_FILENAME)

METR_IMC_EXCEL_PATH = os.path.join(RAW_MISCELLANEOUS_DIR, TRAFFIC_EXCEL_FILENAME)
METR_IMC_EXCEL_TRAINING_PATH = os.path.join(
    RAW_MISCELLANEOUS_DIR, TRAFFIC_TRAINING_EXCEL_FILENAME
)
METR_IMC_EXCEL_TEST_PATH = os.path.join(
    RAW_MISCELLANEOUS_DIR, TRAFFIC_TEST_EXCEL_FILENAME
)

SENSOR_IDS_PATH = os.path.join(RAW_DATASET_ROOT_DIR, SENSOR_IDS_FILENAME)
METADATA_PATH = os.path.join(RAW_DATASET_ROOT_DIR, METADATA_FILENAME)
SENSOR_LOCATIONS_PATH = os.path.join(RAW_DATASET_ROOT_DIR, SENSOR_LOCATIONS_FILENAME)
DISTANCES_PATH = os.path.join(RAW_DATASET_ROOT_DIR, DISTANCES_FILENAME)
ADJ_MX_PATH = os.path.join(RAW_DATASET_ROOT_DIR, ADJ_MX_FILENAME)


# Todo: 주석 해제
# Todo: 상수에 의존하지 않게 변경

def run_process():
    os.makedirs(RAW_MISCELLANEOUS_DIR, exist_ok=True)

    download_nodelink_raw()
    collect_imcrts()
    generate_metr_imc()
    split_traffic_data()
    build_dataset()


def download_nodelink_raw():
    logger.info("Downloading Node-Link Data...")
    nodelink_raw_path = download_nodelink(NODELINK_TARGET_DIR, NODELINK_RAW_URL)
    nodelink_data = NodeLink(nodelink_raw_path).filter_by_gu_codes(INCHEON_REGION_CODES)
    nodelink_data.export(NODELINK_TARGET_DIR)
    logger.info("Downloading Done")


def collect_imcrts():
    logger.info("Collecting IMCRTS Data...")
    collector = IMCRTSCollector(
        api_key=PDP_KEY,
        start_date=IMCRTS_START_DATE,
        end_date=IMCRTS_END_DATE,
    )
    collector.collect(ignore_empty=True)
    collector.to_pickle(output_dir=IMCRTS_DIR, file_name=IMCRTS_FILENAME)
    # collector.to_excel(output_dir=IMCRTS_DIR)
    logger.info("Collecting Done")


def generate_metr_imc(
    road_data_path: str = ROAD_DATA_PATH,
    traffic_data_path: str = IMCRTS_PATH,
):
    road_data: gpd.GeoDataFrame = gpd.read_file(road_data_path)
    traffic_data = TrafficData.import_from_pickle(traffic_data_path)

    logger.info("Matching Link IDs...")
    traffic_data.sensor_filter = road_data["LINK_ID"].tolist()
    traffic_data.to_hdf(METR_IMC_PATH)
    # traffic_data.to_excel(METR_IMC_EXCEL_PATH)
    logger.info("Matching Done")


def split_traffic_data(
    traffic_data_path: str = METR_IMC_PATH,
):
    logger.info("Splitting Dataset...")
    traffic_data = TrafficData.import_from_hdf(traffic_data_path)

    traffic_data.start_time = TRAINING_START_DATETIME
    traffic_data.end_time = TRAINING_END_DATETIME
    training_set = set(traffic_data.data.columns)
    traffic_data.to_hdf(METR_IMC_TRAINING_PATH)
    # traffic_data.to_excel(METR_IMC_EXCEL_TRAINING_PATH)

    traffic_data.reset_data()
    traffic_data.start_time = TEST_START_DATETIME
    traffic_data.end_time = TEST_END_DATETIME
    test_set = set(traffic_data.data.columns)
    test_target = training_set & test_set
    logger.info("Training-Test Difference: ", len(test_set - training_set))
    traffic_data.sensor_filter = list(test_target)

    traffic_data.to_hdf(METR_IMC_TEST_PATH)
    # traffic_data.to_excel(METR_IMC_EXCEL_TEST_PATH)
    logger.info("Splitting Done")


def build_dataset(
    training_path: TrafficData = METR_IMC_TRAINING_PATH,
    metadata_path: str = METADATA_PATH,
    sensor_locations_path: str = SENSOR_LOCATIONS_PATH,
    distances_path: str = DISTANCES_PATH,
    adj_mx_path: str = ADJ_MX_PATH,
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
