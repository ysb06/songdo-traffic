import logging
import os

from metr.dataset.imcrts.collector import IMCRTSCollector
from metr.dataset.metr_imc.generator import MetrImcDatasetGenerator
from metr.dataset.nodelink.converter import INCHEON_CODE, NodeLink
from metr.dataset.utils import download_file, extract_zip_file

from .conftest import Configs
from metr.components import Metadata


def test_environment_variables():
    PDP_KEY = os.environ.get("PDP_KEY")
    print(PDP_KEY)
    assert PDP_KEY is not None


def test_nodelink_data_downloading(configs: Configs):
    target_dir = configs.NODELINK_TARGET_DIR
    print("Node-Link downloaded to", target_dir)

    nodelinke_raw_file_path = download_file(configs.NODELINK_DATA_URL, target_dir)
    print("Extracting NODE-LINK Data...")
    nodelink_raw_data_path = extract_zip_file(nodelinke_raw_file_path, target_dir)
    print(f"Extracted at {nodelink_raw_data_path}")
    NodeLink(nodelink_raw_data_path).filter_by_gu_codes(INCHEON_CODE).export(target_dir)


def test_imcrts_data_collecting(configs: Configs):
    target_dir = configs.IMCRTS_TARGET_DIR
    start_date = configs.target_start_date
    end_date = configs.target_end_date
    print("IMCRTS downloaded to", target_dir)

    PDP_KEY = os.environ.get("PDP_KEY")
    collector = IMCRTSCollector(key=PDP_KEY, start_date=start_date, end_date=end_date)
    collector.collect()
    collector.to_pickle(output_dir=target_dir)
    collector.to_excel(output_dir=target_dir)


def test_dataset_generation(configs: Configs):
    nodelink_dir = configs.NODELINK_TARGET_DIR
    imcrts_dir = configs.IMCRTS_TARGET_DIR
    target_dir = configs.target_dir
    print("Metr-IMC generated to", target_dir)

    MetrImcDatasetGenerator(nodelink_dir, imcrts_dir).generate(target_dir)

def test_metadata_generation(configs: Configs):
    metadata = Metadata.import_from_nodelink(configs.NODELINK_TARGET_DIR)

    hdf_path = os.path.join(configs.target_dir, "metadata.h5")
    xls_path = os.path.join(configs.target_dir, "miscellaneous", "metadata.xlsx")

    metadata.to_hdf(hdf_path)
    metadata.to_excel(xls_path)

    assert os.path.exists(hdf_path)
    assert os.path.exists(xls_path)