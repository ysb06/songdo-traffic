import glob
import logging
import os
import random
from datetime import datetime

from metr.components.metr_imc import TrafficData
import yaml

from .best_interpolation_search import train_traffic_model
from ..preprocessing.missing import process_missing
from ..preprocessing.outlier import process_outlier
from ..utils import fix_seed

TARGET_DATA_PATH = "./output/all_processed"

logger = logging.getLogger(__name__)


def do_experiment(seed=47):
    fix_seed(seed)

    targets = glob.glob(f"{TARGET_DATA_PATH}/*.h5", recursive=True)
    training_id = datetime.now().strftime("%Y%m%d%H%M%S")
    for i in range(5):
        train_traffic_model(
            training_id=training_id,
            model_postfix=f"1_{i}",
            target_sensor_idx=0,
            seq_length=24,
        )
    for i in range(5):
        train_traffic_model(
            training_id=training_id,
            model_postfix=f"3_{i}",
            target_sensor_idx=0,
            seq_length=24 * 3,
        )
    for i in range(5):
        train_traffic_model(
            training_id=training_id,
            model_postfix=f"7_{i}",
            target_sensor_idx=0,
            seq_length=24 * 7,
        )

    targets = glob.glob(
        f"./output/predictions/abs_cap-linear/abs_cap-linear/target_0000/*.yaml",
        recursive=True,
    )
    results = {}
    for target in targets:
        key = target.split("_")[-2]
        with open(target, "r") as f:
            result = yaml.safe_load(f)

        if key not in results:
            results[key] = result["test_mae"]
        else:
            results[key] += result["test_mae"]

    print(results)
