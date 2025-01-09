import glob
import logging
import os
import random

from metr.components.metr_imc import TrafficData

from .main import train_traffic_model
from .preprocessing.missing import process_missing
from .preprocessing.outlier import process_outlier

TARGET_DATA_PATH = "./output/all_processed"

logger = logging.getLogger(__name__)

process_outlier()
process_missing()

sample_rate = 0.1

targets = glob.glob(f"{TARGET_DATA_PATH}/*.h5", recursive=True)
for target in targets:
    logger.info(f"Training model with {target}")
    raw = TrafficData.import_from_hdf(target)
    sensor_list = raw.data.columns

    target_sensors = random.sample(range(len(sensor_list)), round(len(sensor_list) * sample_rate))

    for idx in target_sensors:
        train_traffic_model(
            traffic_data_path=target,
            target_sensor_idx=idx,
        )
