import argparse
import os

from .training import train
from .utils import get_config


def run():
    parser = argparse.ArgumentParser(description="STGCN_WAVE")
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

    config.lr = args.lr if args.lr is not None else config.lr
    config.disablecuda = (
        args.disablecuda if args.disablecuda is not None else config.disablecuda
    )
    config.batch_size = (
        args.batch_size if args.batch_size is not None else config.batch_size
    )
    config.epochs = args.epochs if args.epochs is not None else config.epochs
    config.num_layers = (
        args.num_layers if args.num_layers is not None else config.num_layers
    )
    config.window = args.window if args.window is not None else config.window
    config.sensorsfilepath = (
        args.sensorsfilepath
        if args.sensorsfilepath is not None
        else config.sensorsfilepath
    )
    config.disfilepath = (
        args.disfilepath if args.disfilepath is not None else config.disfilepath
    )
    config.tsfilepath = (
        args.tsfilepath if args.tsfilepath is not None else config.tsfilepath
    )
    config.savemodelpath = (
        args.savemodelpath if args.savemodelpath is not None else config.savemodelpath
    )
    config.pred_len = args.pred_len if args.pred_len is not None else config.pred_len
    config.control_str = (
        args.control_str if args.control_str is not None else config.control_str
    )
    config.channels = args.channels if args.channels is not None else config.channels
    config.seed = args.seed if args.seed is not None else config.seed

    train(config)

if __name__ == "__main__":
    run()