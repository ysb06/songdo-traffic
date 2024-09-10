import logging
from argparse import ArgumentParser
import os

from .collector import IMCRTSCollector, load_key
from . import PDP_KEY, IMCRTS_DEFAULT_DIR

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
    default=IMCRTS_DEFAULT_DIR,
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
    api_key = PDP_KEY
else:
    api_key = load_key(args.key)

collector = IMCRTSCollector(
    key=api_key,
    start_date=args.start_date,
    end_date=args.end_date,
)
collector.collect()
collector.to_pickle(args.output_dir)
collector.to_excel(args.output_dir)
