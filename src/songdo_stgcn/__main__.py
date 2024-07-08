import argparse
import logging

from .utils import get_torch_device, set_env

from .config import load_config
import pprint


logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="STGCN")
parser.add_argument(
    "--config",
    default="./configs/stgcn/config_base.yaml",
    help="Config path for training and testing STGCN model",
)
args = parser.parse_args()

config = load_config(args.config)
logger.info(f"Loaded configuration from {args.config}:\n\n{pprint.pformat(config)}")

# Setting environment
set_env(config.seed)
torch_device = get_torch_device(config.device)
Ko = config.n_his - (config.Kt - 1) * 2 * config.stblock_num
