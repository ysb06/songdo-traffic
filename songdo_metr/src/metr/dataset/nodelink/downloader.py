import os
import logging

from metr.dataset.utils import download_file, extract_zip_file

NODELINK_20231228_DATA_URL = (
    "https://www.its.go.kr/opendata/nodelinkFileSDownload/DF_193/0"
)

logger = logging.getLogger(__name__)


def download_nodelink(
    url: str = NODELINK_20231228_DATA_URL,
    download_dir: str = "./datasets/metr-imc/nodelink",
    chunk_size: int = 1024,
):
    os.makedirs(download_dir, exist_ok=True)
    nodelinke_raw_file_path = download_file(url, download_dir, chunk_size)
    logger.info("Extracting NODE-LINK Data...")
    nodelink_raw_data_path = extract_zip_file(nodelinke_raw_file_path, download_dir)
    logger.info(f"Extracted at {nodelink_raw_data_path}")

    return nodelink_raw_data_path
