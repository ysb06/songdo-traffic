import os
import logging

from metr.dataset.utils import download_file, extract_zip_file
from . import NODELINK_DATA_URL, NODELINK_DEFAULT_DIR

logger = logging.getLogger(__name__)


def download_nodelink(
    url: str = NODELINK_DATA_URL,
    download_dir: str = NODELINK_DEFAULT_DIR,
    download_chunk_size: int = 1024,
):
    os.makedirs(download_dir, exist_ok=True)
    logger.info("Downloading...")
    nodelinke_raw_file_path = download_file(url, download_dir, download_chunk_size)
    logger.info("Extracting NODE-LINK Data...")
    nodelink_raw_data_dir = extract_zip_file(nodelinke_raw_file_path, download_dir)
    logger.info(f"Extracted at {nodelink_raw_data_dir}")

    return nodelink_raw_data_dir
