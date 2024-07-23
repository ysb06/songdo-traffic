import logging
from argparse import ArgumentParser

from .generator import MetrImcDatasetGenerator

logging.basicConfig(
    format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
    datefmt="%Y/%m/%d %H:%M:%S",
    level=logging.INFO,
)

parser = ArgumentParser()
parser.add_argument(
    "--nodelink_dir",
    help="Path for the node-link folder",
    default="./datasets/metr-imc/nodelink",
)
parser.add_argument(
    "--imcrts_dir",
    help="Path for the IMCRTS folder",
    default="./datasets/metr-imc/imcrts",
)
parser.add_argument(
    "--output_dir",
    help="Path for the output folder",
    default="./datasets/metr_imc",
)
args = parser.parse_args()
MetrImcDatasetGenerator(args.nodelink_dir, args.imcrts_dir).generate(args.output_dir)
