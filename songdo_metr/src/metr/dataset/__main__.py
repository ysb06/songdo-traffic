import logging
import os

from .imcrts.collector import IMCRTSCollector
from .metr_imc.generator import MetrImcDatasetGenerator
from .nodelink.converter import INCHEON_CODE, NodeLink
from .utils import download_file, extract_zip_file

logger = logging.getLogger(__name__)



NODELINK_DATA_URL = "https://www.its.go.kr/opendata/nodelinkFileSDownload/DF_193/0"
NODELINK_DIR_PATH = "../datasets/metr-imc/nodelink"

IMCRTS_DIR_PATH = "../datasets/metr-imc/imcrts"
PDP_KEY = os.environ.get("PDP_KEY")

METR_IMC_PATH = "../datasets/metr-imc"

# Generate NodeLink Data
nodelinke_raw_file_path = download_file(NODELINK_DATA_URL, NODELINK_DIR_PATH)
logger.info("Extracting NODE-LINK Data...")
nodelink_raw_data_path = extract_zip_file(nodelinke_raw_file_path, NODELINK_DIR_PATH)
logger.info(f"Extracted at {nodelink_raw_data_path}")
NodeLink(nodelink_raw_data_path).filter_by_gu_codes(INCHEON_CODE).export(
    NODELINK_DIR_PATH
)

# Generate IMCRTS Data
IMCRTSCollector(
    key=PDP_KEY,
    start_date="20230101",
    end_date="20240301",
).collect(output_dir=IMCRTS_DIR_PATH)


# Generate Metr-IMC Data
MetrImcDatasetGenerator(NODELINK_DIR_PATH, IMCRTS_DIR_PATH).generate(METR_IMC_PATH)
