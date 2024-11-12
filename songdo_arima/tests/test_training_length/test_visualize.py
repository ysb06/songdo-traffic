import pickle
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd
from pmdarima import ARIMA

from songdo_arima.utils import HyperParams
from metr.components import TrafficData


def test_visualize():
    output_root_dir = "./output/arima-01-01-short"

    with open(f"{output_root_dir}/results.pkl", "rb") as f:
        results: Dict = pickle.load(f)
    
    print("Results:")
    for week, result in results.items():
        if result["is_best_mae"] or result["is_best_rmse"]:
            print(week, "-->")
            print(result)

    with open(f"{output_root_dir}/best_mae_model_1630008100.pkl", "rb") as f:
        best_mae_model: ARIMA = pickle.load(f)

    with open(f"{output_root_dir}/best_rmse_model_1630008100.pkl", "rb") as f:
        best_rmse_model: ARIMA = pickle.load(f)

    with open(f"{output_root_dir}/last_model_1630008100_11.pkl", "rb") as f:
        last_model: ARIMA = pickle.load(f)

    test_raw = pd.read_hdf("../datasets/metr-imc/metr-imc-test.h5")
    test_target_data = test_raw["1630008100"]

    print("Test:", test_target_data.index.min(), "to", test_target_data.index.max())

    best_mae_pred_result: pd.Series = best_mae_model.predict(
        n_periods=len(test_target_data)
    )
    best_mae_pred_result.index = test_target_data.index

    best_rmse_pred_result: pd.Series = best_rmse_model.predict(
        n_periods=len(test_target_data)
    )
    best_rmse_pred_result.index = test_target_data.index

    last_pred_result: pd.Series = last_model.predict(n_periods=len(test_target_data))
    last_pred_result.index = test_target_data.index

    

    plt.subplot(3, 1, 1)
    plt.plot(test_target_data, label="True")
    plt.plot(best_mae_pred_result, label="Best MAE")
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(test_target_data, label="True")
    plt.plot(best_rmse_pred_result, label="Best RMSE")
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(test_target_data, label="True")
    plt.plot(last_pred_result, label="Last")
    plt.legend()

    plt.show()


# 결과는 13일이 가장 좋은 결과를 보임