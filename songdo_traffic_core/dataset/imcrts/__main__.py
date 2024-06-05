import logging
from argparse import ArgumentParser

from .collector import IMCRTSCollector, load_key

logging.basicConfig(
    format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
    datefmt="%Y/%m/%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

parser = ArgumentParser()
parser.add_argument(
    "--key",
    help="Path for the key file",
    default="./datasets/imcrts/key",
)
parser.add_argument(
    "--date_range",
    help="Date range for the collecting data (format: YYYYMMDD-YYYYMMDD)",
    default="20230101-20231231",
)
parser.add_argument(
    "--output_dir",
    help="Path to output directory",
    default="./datasets/imcrts/",
)

args = parser.parse_args()
start_date, end_date = args.date_range.split("-")
api_key = load_key(args.key)

IMCRTSCollector(
    key=api_key, start_date=start_date, end_date=end_date, output_dir=args.output_dir
).collect()