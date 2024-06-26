import logging
import os
from argparse import ArgumentParser

import pandas as pd

from .converter import INCHEON_CODE, NodeLink, get_sensor_node_list

logging.basicConfig(
    format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
    datefmt="%Y/%m/%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

parser = ArgumentParser()
parser.add_argument(
    "--nodelink_dir",
    help="Path to the Node Link data directory",
    default="./datasets/imc_nodelink_2024/raws/[2024-02-23]NODELINKDATA",
)
parser.add_argument(
    "--songdo_traffic_file",
    help="Path to the Songdo Traffic data file",
    default="./datasets/imc_nodelink_2024/raws/ICT_ICTSPOTDTCT5M0000001_2024_02.csv",
)
parser.add_argument(
    "--output_dir",
    help="Path to output directory",
    default="./datasets/imc_nodelink_2024/",
)
args = parser.parse_args()

logger.info("Loading NodeLink datasets...")
nodelink_data = NodeLink(args.nodelink_dir)
logger.info("Datasets loaded.")
imc_nodelink = nodelink_data.filter_by_gu_codes(INCHEON_CODE)
imc_nodelink.export(args.output_dir)

logger.info("Generating Songdo Sensor data...")
try:
    traffic_data = pd.read_csv(args.songdo_traffic_file)
    sensor_list = get_sensor_node_list(traffic_data)
    sensor_list.columns = ["SNSR_ID", "SNSR_NAME", "LOCA_NAME", "AREA_NAME", "geometry"]
    sensor_list.to_file(os.path.join(args.output_dir, "sensor_node.shp"), encoding="utf-8")
except:
    logger.error("Failed to generate Songdo Sensor data.")
    logger.error("Skipping...")

logger.info("Converting Completed.")
