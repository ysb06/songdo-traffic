import logging
from argparse import ArgumentParser

from .generator import MetrImcDatasetGenerator
from ..nodelink import NODELINK_DEFAULT_DIR
from ..imcrts import IMCRTS_DEFAULT_DIR
from . import METR_IMC_DEFAULT_DIR

parser = ArgumentParser()
parser.add_argument(
    "--nodelink_dir",
    help="Path for the node-link folder",
    default=NODELINK_DEFAULT_DIR,
)
parser.add_argument(
    "--imcrts_dir",
    help="Path for the IMCRTS folder",
    default=IMCRTS_DEFAULT_DIR,
)
parser.add_argument(
    "--output_dir",
    help="Path for the output folder",
    default=METR_IMC_DEFAULT_DIR,
)
args = parser.parse_args()

MetrImcDatasetGenerator(args.nodelink_dir, args.imcrts_dir).generate(args.output_dir)
