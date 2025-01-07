import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import yaml
import textwrap
from pprint import pprint
import pandas as pd

SARIMAX_RESULTS_DIR = "../output/metr-imc-interpolated"


def plot_results_from_pickle():
    targets = glob.glob(f"{SARIMAX_RESULTS_DIR}/**/*.yaml", recursive=True)
    targets = [target for target in targets if "results/" in target]

    results = []
    for target in targets:
        with open(target, "r") as f:
            data = yaml.load(f, Loader=yaml.FullLoader)

        target_splits = target.split("/")
        outlier_type = target_splits[3]
        interpolation_type = target_splits[4]
        sensor_id = target_splits[-1].split(".")[0]

        result = {
            "outlier_type": outlier_type,
            "interpolation_type": interpolation_type,
            "sensor_id": sensor_id,
            "stationary_test_passed": None,
            "prediction_failed_error": None,
            "MAE": data["mae"],
            "RMSE": data["rmse"],
        }
        if "stationary_test_passed" in data:
            result["stationary_test_passed"] = data["stationary_test_passed"]

        if "prediction_failed_error" in data:
            result["prediction_failed_error"] = data["prediction_failed_error"]

        results.append(result)

    results_df = pd.DataFrame(results)
    results_df.to_excel(os.path.join(SARIMAX_RESULTS_DIR, "results.xlsx"))


def plot_results():
    print("Crawling from", os.path.abspath(SARIMAX_RESULTS_DIR))
    print("Found:", os.path.exists(SARIMAX_RESULTS_DIR))
    targets = glob.glob(f"{SARIMAX_RESULTS_DIR}/**/results.yaml", recursive=True)
    pprint(targets)

    mae = {}
    rmse = {}
    for target in targets:
        temp = target.split("/")
        name = temp[-4] + "/" + temp[-3]
        with open(target, "r") as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
            mae[name] = data["mean_MAE"]
            rmse[name] = data["mean_RMSE"]

    x = np.arange(len(mae))
    plt.figure(figsize=(10, 5))
    plt.bar(x - 0.2, mae.values(), width=0.4, label="MAE")
    plt.bar(x + 0.2, rmse.values(), width=0.4, label="RMSE")
    wrapped_labels = [textwrap.fill(label, width=16) for label in mae.keys()]
    plt.xticks(x, wrapped_labels, ha="right")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    plot_results_from_pickle()
