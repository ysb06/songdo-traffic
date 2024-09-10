import logging
from argparse import ArgumentParser

from metr.dataset.nodelink.downloader import download_nodelink

from .converter import INCHEON_CODE, NodeLink
from . import NODELINK_DATA_URL, NODELINK_DEFAULT_DIR

logger = logging.getLogger(__name__)

parser = ArgumentParser()
parser.add_argument(
    "--nodelink_url",
    help="URL for downloading nodelink data",
    default=NODELINK_DATA_URL,
)
parser.add_argument(
    "--output_dir",
    help="Path to output directory",
    default=NODELINK_DEFAULT_DIR,
)
args = parser.parse_args()

nodelink_raw_root = download_nodelink(
    url=args.nodelink_url,
    download_dir=args.output_dir,
)
nodelink = NodeLink(nodelink_raw_root)
imc_nodelink = nodelink.filter_by_gu_codes(INCHEON_CODE)
imc_nodelink.export(NODELINK_DEFAULT_DIR)
