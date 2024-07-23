import logging
import os
from argparse import ArgumentParser

import pandas as pd

from metr.dataset.nodelink.downloader import download_nodelink

from .converter import INCHEON_CODE, NodeLink

logging.basicConfig(
    format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
    datefmt="%Y/%m/%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


parser = ArgumentParser()
parser.add_argument(
    "--nodelink_url",
    help="URL for downloading nodelink data",
    default="https://www.its.go.kr/opendata/nodelinkFileSDownload/DF_193/0",
)
parser.add_argument(
    "--output_dir",
    help="Path to output directory",
    default="./datasets/metr-imc/nodelink",
)
args = parser.parse_args()

nodelink_raw_file_path = download_nodelink(args.nodelink_url, args.output_dir)
nodelink_raw = (
    NodeLink(nodelink_raw_file_path)
    .filter_by_gu_codes(INCHEON_CODE)
    .export(args.output_dir)
)
