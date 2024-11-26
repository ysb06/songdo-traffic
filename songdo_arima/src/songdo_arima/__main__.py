import glob
import os

import yaml
from songdo_arima.sarimax_training import train_model
import logging

logger = logging.getLogger(__name__)

yaml_path = os.path.join("./configs", "config.yaml")
with open(yaml_path, "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

data_root_dir = config["traffic_data_root_dir"]

query = os.path.join(data_root_dir, "**")
glob_objs = glob.glob(query, recursive=True)
for glob_obj in glob_objs:
    if "metr-imc.h5" in glob_obj:
        logger.info(f"Training model for {glob_obj}")
        train_model(glob_obj, config)
        break