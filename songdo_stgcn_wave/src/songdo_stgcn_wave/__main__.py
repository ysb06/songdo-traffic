import argparse
import os
import logging

from .training import train, train_new
from .utils import get_config
from .test import test_model

logger = logging.getLogger(__name__)


def run():
    parser = argparse.ArgumentParser(description="STGCN_WAVE")
    parser.add_argument(
        "--training_only", action="store_true", help="Run Model Training Only"
    )
    parser.add_argument(
        "--test_only", action="store_true", help="Run Model Training Only"
    )
    parser.add_argument(
        "--config", default="base", type=str, help="Config file for hyper-parameters"
    )
    parser.add_argument("--lr", type=float, help="learning rate")
    parser.add_argument("--disablecuda", action="store_true", help="Disable CUDA")
    parser.add_argument(
        "--batch_size", type=int, help="batch size for training and validation"
    )
    parser.add_argument("--epochs", type=int, help="epochs for training")
    parser.add_argument("--num_layers", type=int, help="number of layers")
    parser.add_argument("--window", type=int, help="window length")
    parser.add_argument("--sensorsfilepath", type=str, help="sensors file path")
    parser.add_argument("--disfilepath", type=str, help="distance file path")
    parser.add_argument("--tsfilepath", type=str, help="ts file path")
    parser.add_argument("--savemodelpath", type=str, help="save model path")
    parser.add_argument(
        "--pred_len", type=int, help="how many steps away we want to predict"
    )
    parser.add_argument("--control_str", type=str, help="model structure controller")
    parser.add_argument(
        "--channels", type=int, nargs="+", help="model structure controller"
    )
    parser.add_argument("--seed", type=int, help="seed for training")

    args = parser.parse_args()

    config_path = os.path.join("configs", f"{args.config}.yaml")
    config = get_config(config_path)

    for key, value in vars(args).items():
        if value is not None:
            setattr(config, key, value)

    if not args.test_only:
        logger.info(f"Start Training Model with:\r\n{config}")
        train_new(config)

    if not args.training_only:
        logger.info(f"Start Testing Model with:\r\n{config}")
        test_model(config)


if __name__ == "__main__":
    run()
