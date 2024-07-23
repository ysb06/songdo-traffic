import logging
from argparse import ArgumentParser
import os

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
    help="Path for the key file (Default: Using PDP_KEY environment variable)",
    type=str,
    default=None,
)

parser.add_argument(
    "--output_dir",
    help="Path to output directory",
    type=str,
    default="./datasets/imcrts/",
)

parser.add_argument(
    "--start_date",
    help="Start for the collecting data (format: YYYYMMDD)",
    type=str,
    default="20230101",
)

parser.add_argument(
    "--end_date",
    help="End date for the collecting data (format: YYYYMMDD)",
    type=str,
    default="20231231",
)

args = parser.parse_args()
if args.key is None:
    api_key = os.environ.get("PDP_KEY")
else:
    api_key = load_key(args.key)

IMCRTSCollector(
    key=api_key,
    start_date=args.start_date,
    end_date=args.end_date,
    output_dir=args.output_dir,
).collect()
