import os
import pickle
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd
from metr.components import TrafficData
from pmdarima import ARIMA, auto_arima
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from songdo_arima.utils import HyperParams


def test_training_short(configs: HyperParams):
    output_root_dir = "./output/arima-01-01-short"
    os.makedirs(output_root_dir, exist_ok=True)

    raw = TrafficData.import_from_hdf(configs.traffic_training_data_path)
    raw_data = raw.data
    target_sensor = "1630008100"
    data = raw_data[target_sensor]
    print("Cropped:", data.index.min(), "to", data.index.max())

    test_raw = pd.read_hdf("../datasets/metr-imc/metr-imc-test.h5")
    test_data = test_raw[target_sensor]

    weeks = 12
    result: Dict[int, Dict] = {}
    best_mae_week = -1
    best_mae_week_model: ARIMA = None
    best_mae = 1e9
    best_rmse_week = -1
    best_rmse_week_model: ARIMA = None
    best_rmse = 1e9
    last_model: ARIMA = None
    for week in range(weeks):
        print("Week:", week + 1)
        length = 7 * 24 * (week + 1)
        cropped_data = data.iloc[-length:]
        model = train_arima(cropped_data)
        mae, rmse = do_testing_arima(model, test_data)

        result[week] = {
            "is_best_mae": False,
            "is_best_rmse": False,
            "params": model.get_params(),
            "mae": mae,
            "rmse": rmse,
        }

        if mae < best_mae:
            best_mae = mae
            best_mae_week = week
            best_mae_week_model = model
            print("Best MAE:", best_mae, "Week:", week)

        if rmse < best_rmse:
            best_rmse = rmse
            best_rmse_week = week
            best_rmse_week_model = model
            print("Best RMSE:", best_rmse, "Week:", week)
        
        last_model = model

    result[best_mae_week]["is_best_mae"] = True
    result[best_rmse_week]["is_best_rmse"] = True

    with open(f"{output_root_dir}/results.pkl", "wb") as f:
        pickle.dump(result, f)

    with open(f"{output_root_dir}/best_mae_model_{target_sensor}.pkl", "wb") as f:
        pickle.dump(best_mae_week_model, f)

    with open(f"{output_root_dir}/best_rmse_model_{target_sensor}.pkl", "wb") as f:
        pickle.dump(best_rmse_week_model, f)
    
    with open(f"{output_root_dir}/last_model_{target_sensor}_{week}.pkl", "wb") as f:
        pickle.dump(last_model, f)


def train_arima(data: pd.Series) -> ARIMA:
    model: ARIMA = auto_arima(
        data,
        start_p=0,
        max_p=3,
        start_q=0,
        max_q=3,
        d=None,  # 'd' 값을 자동으로 결정
        start_P=0,
        max_P=3,
        start_Q=0,
        max_Q=3,
        D=None,  # 'D' 값을 자동으로 결정
        m=24,  # 계절 주기 (시간별 데이터의 일일 계절성)
        seasonal=True,
        stepwise=True,  # 스텝와이즈 알고리즘 사용으로 계산 시간 단축
        suppress_warnings=True,
        error_action="ignore",
        trace=True,  # 진행 상황을 보려면 True로 설정
    )

    return model


def do_testing_arima(model: ARIMA, test_data: pd.Series):
    pred_result: pd.Series = model.predict(n_periods=len(test_data))

    mae = mean_absolute_error(test_data, pred_result)
    rmse = root_mean_squared_error(test_data, pred_result)

    return mae, rmse
