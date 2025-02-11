from collections import defaultdict
import glob
from typing import DefaultDict, Dict, Union

import pandas as pd
import yaml

RESULTS_DIR = "./output/predictions_evaluation"


def test_visualize_by_sensor():
    results_dir_list = glob.glob(f"{RESULTS_DIR}/*", recursive=True)

    mae_results: DefaultDict[int, Dict[str, float]] = defaultdict(dict)
    rmse_results: DefaultDict[int, Dict[str, float]] = defaultdict(dict)
    smape_results: DefaultDict[int, Dict[str, float]] = defaultdict(dict)
    mape_results: DefaultDict[int, Dict[str, float]] = defaultdict(dict)
    sensor_names: Dict[int, str] = {}

    for result_dir in results_dir_list:
        method_name = result_dir.split("/")[-1]
        metrics_file_paths = glob.glob(f"{result_dir}/*.yaml", recursive=True)

        for metric_file_path in metrics_file_paths:
            sensor_fullname = metric_file_path.split("/")[-1].split(".")[0]
            sensor_idx = int(sensor_fullname.split("_")[1])
            sensor_name = sensor_fullname.split("_")[2]

            sensor_names[sensor_idx] = sensor_name

            with open(metric_file_path, "r") as f:
                metrics = yaml.safe_load(f)

                mae_results[sensor_idx][method_name] = metrics["MAE"]
                rmse_results[sensor_idx][method_name] = metrics["RMSE"]
                smape_results[sensor_idx][method_name] = metrics["sMAPE"]
                mape_results[sensor_idx][method_name] = metrics["MAPE (Only Non-Zero)"]

    mae_df = pd.DataFrame(mae_results)
    rmse_df = pd.DataFrame(rmse_results)
    smape_df = pd.DataFrame(smape_results)
    mape_df = pd.DataFrame(mape_results)

    mae_df.to_excel("./output/mae_results.xlsx")
    rmse_df.to_excel("./output/rmse_results.xlsx")
    smape_df.to_excel("./output/smape_results.xlsx")
    mape_df.to_excel("./output/mape_results.xlsx")


def test_cheet():
    mae_cheet_df = pd.read_excel("./output/mae_results_cheet_raw.xlsx", sheet_name="Cheet 1", index_col=0, header=0)
    target_sensors = [idx for idx in mae_cheet_df.columns if type(idx) == int]

    mae_df = pd.read_excel("./output/mae_results.xlsx")
    rmse_df = pd.read_excel("./output/rmse_results.xlsx")
    smape_df = pd.read_excel("./output/smape_results.xlsx")
    mape_df = pd.read_excel("./output/mape_results.xlsx")

    mae_df[target_sensors].to_excel("./output/cheet_mae_results.xlsx")
    rmse_df[target_sensors].to_excel("./output/cheet_rmse_results.xlsx")
    smape_df[target_sensors].to_excel("./output/cheet_smape_results.xlsx")
    mape_df[target_sensors].to_excel("./output/cheet_mape_results.xlsx")
