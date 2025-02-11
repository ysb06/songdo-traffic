import glob
import logging
import os
from collections import defaultdict
from typing import List

import numpy as np
import torch
import yaml
from metr.components.metr_imc import TrafficData
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from torch import Tensor
from tqdm import tqdm

from ..dataset import TrafficDataModule
from ..model import SongdoTrafficLightningModel
from ..plot import plot_results_metrics
from ..utils import (fix_seed, non_zero_mape,
                     symmetric_mean_absolute_percentage_error)

RAW_DATA_PATH = "../datasets/metr-imc/metr-imc.h5"
SYNC_PROCESSED_DIR = "./output/sync_processed"
PREDICTIONS_DIR = "./output/predictions"
OUTPUT_DIR = "./output/predictions_evaluation"
os.makedirs(OUTPUT_DIR, exist_ok=True)

logger = logging.getLogger(__name__)


def do_evaluation(seed=42):
    fix_seed(seed)

    synced_targets = glob.glob(f"{SYNC_PROCESSED_DIR}/*.h5", recursive=True)

    for synced_target_path in synced_targets:
        synced_target_result_name = synced_target_path.split("/")[-1].split(".")[0]
        synced_target_data = TrafficData.import_from_hdf(synced_target_path).data
        pred_results_dir = f"{PREDICTIONS_DIR}/{synced_target_result_name}"
        eval_results_dir = f"{OUTPUT_DIR}/{synced_target_result_name}"
        os.makedirs(eval_results_dir, exist_ok=True)

        pred_results = glob.glob(f"{pred_results_dir}/*", recursive=True)
        for pred_result_dir in pred_results:
            pred_model_path = glob.glob(f"{pred_result_dir}/*.ckpt", recursive=True)[0]
            pred_result_name = pred_result_dir.split("/")[-1].split(".")[0]
            sensor_idx = int(pred_result_name.split("_")[-1])

            target_sensor_name = synced_target_data.columns[sensor_idx]

            raw_data_module = TrafficDataModule(
                traffic_data_path=RAW_DATA_PATH,
                target_traffic_sensor=sensor_idx,
                batch_size=512,
            )
            raw_data_module.setup()
            raw_dataloader = raw_data_module.predict_dataloader()

            pred_model = SongdoTrafficLightningModel.load_from_checkpoint(
                checkpoint_path=pred_model_path
            )
            pred_model.eval()

            test_scaled_true: List[np.ndarray] = []
            test_scaled_pred: List[np.ndarray] = []
            for idx, item in tqdm(enumerate(raw_dataloader), total=len(raw_dataloader)):
                x: Tensor = item[0]
                y: Tensor = item[1]

                x_nan_mask = torch.isnan(x.view(x.size(0), -1)).any(dim=1)
                y_nan_mask = torch.isnan(y.view(y.size(0), -1)).any(dim=1)
                invalid_mask = (
                    x_nan_mask | y_nan_mask
                )  # x 또는 y 둘 중 하나라도 NaN이 있으면 invalid
                valid_mask = ~invalid_mask

                x_filtered = x[valid_mask]
                y_filtered = y[valid_mask]

                if x.size(0) != x_filtered.size(0):
                    print(
                        f"[{idx + 1}/{len(raw_dataloader)}] Batch Filtered: {x.size(0)} -> {x_filtered.size(0)}"
                    )
                if x_filtered.size(0) == 0:
                    print(f"[{idx + 1}/{len(raw_dataloader)}] Batch Passed")
                    continue

                x_filtered = x_filtered.to(pred_model.device)
                y_filtered = y_filtered.to(pred_model.device)

                y_hat: Tensor = pred_model(x_filtered)

                test_scaled_true.append(y_filtered.cpu().detach().numpy())
                test_scaled_pred.append(y_hat.cpu().detach().numpy())

            if len(test_scaled_true) == 0 or len(test_scaled_pred) == 0:
                logger.warning(f"No valid data for sensor {sensor_idx}")
                continue

            test_scaled_true_arr = np.concatenate(test_scaled_true, axis=0)
            test_scaled_pred_arr = np.concatenate(test_scaled_pred, axis=0)
            scaler = raw_data_module.scaler
            test_true_arr = scaler.inverse_transform(test_scaled_true_arr)
            test_pred_arr = scaler.inverse_transform(test_scaled_pred_arr)
            test_true = test_true_arr.squeeze(1)
            test_pred = test_pred_arr.squeeze(1)

            test_mae = mean_absolute_error(test_true, test_pred)
            test_rmse = root_mean_squared_error(test_true, test_pred)
            test_mape, test_mape_zero_excluded = non_zero_mape(test_true, test_pred)
            test_smape = symmetric_mean_absolute_percentage_error(test_true, test_pred)

            logger.info(
                f"MAE: {test_mae:.6f}, RMSE: {test_rmse:.6f}, MAPE: {test_mape:.6f}, sMAPE(0-200): {test_smape:.6f}"
            )

            metrics_dict = {
                "MAE": test_mae,
                "RMSE": test_rmse,
                "MAPE (Only Non-Zero)": test_mape,
                "Excluded Zero in MAPE": test_mape_zero_excluded,
                "sMAPE": test_smape,
            }

            metrics_file_name = f"metrics_{sensor_idx:04d}_{target_sensor_name}.yaml"
            yaml_path = os.path.join(eval_results_dir, metrics_file_name)
            with open(yaml_path, "w") as f:
                yaml.safe_dump(metrics_dict, f)
            logger.info(f"Metrics saved to {yaml_path}")


def plot_metrics():
    results_dir_list = glob.glob(f"{OUTPUT_DIR}/*", recursive=True)

    mae_results = defaultdict(list)
    rmse_results = defaultdict(list)
    smape_results = defaultdict(list)
    mape_results = defaultdict(list)
    
    for result_dir in results_dir_list:
        result_name = result_dir.split("/")[-1]
        metrics_file_paths = glob.glob(f"{result_dir}/*.yaml", recursive=True)

        for metric_file_path in metrics_file_paths:
            with open(metric_file_path, "r") as f:
                metrics = yaml.safe_load(f)

            mae_results[result_name].append(metrics["MAE"])
            rmse_results[result_name].append(metrics["RMSE"])
            smape_results[result_name].append(metrics["sMAPE"])
            mape_results[result_name].append(metrics["MAPE (Only Non-Zero)"])
        
    plot_results_metrics(mae_results, "MAE")
    plot_results_metrics(rmse_results, "RMSE")
    plot_results_metrics(smape_results, "sMAPE")
    plot_results_metrics(mape_results, "MAPE (Only Non-Zero)")