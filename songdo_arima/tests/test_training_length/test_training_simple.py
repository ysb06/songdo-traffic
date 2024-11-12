import os
import pickle
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd
from metr.components import TrafficData
from pmdarima import ARIMA, auto_arima
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from songdo_arima.utils import HyperParams
from songdo_arima.sarimax_training import BestARIMA


def test_training_short(configs: HyperParams):
    output_root_dir = "./output/arima-01-01-simple"
    os.makedirs(output_root_dir, exist_ok=True)

    raw = TrafficData.import_from_hdf(configs.traffic_training_data_path)
    # raw.start_time = pd.Timestamp("2024-03-01 00:00:00")
    raw.start_time = pd.Timestamp("2024-05-01 00:00:00")
    target_sensor = "1630008100"
    train_data = raw.data[target_sensor]
    test_raw = pd.read_hdf("../datasets/metr-imc/metr-imc-test.h5")
    test_data = test_raw[target_sensor]

    model = train_arima(train_data)
    mae, rmse = do_testing_arima(model, test_data)
    print(f"Target: {target_sensor}, MAE: {mae}, RMSE: {rmse}")
    best_model = BestARIMA(model)
    best_model.result = {"Target": target_sensor, "MAE": mae, "RMSE": rmse}
    
    with open(f"{output_root_dir}/last_model_{target_sensor}.pkl", "wb") as f:
        pickle.dump(best_model, f)


def train_arima(data: pd.Series) -> ARIMA:
    model: ARIMA = auto_arima(
        data,
        start_p=0,
        max_p=7,
        start_q=0,
        max_q=7,
        d=None,  # 'd' 값을 자동으로 결정
        start_P=0,
        max_P=7,
        start_Q=0,
        max_Q=7,
        D=None,  # 'D' 값을 자동으로 결정
        m=24,  # 계절 주기 (시간별 데이터의 일일 계절성)
        seasonal=True,
        stepwise=True,  # 스텝와이즈 알고리즘 사용으로 계산 시간 단축
        suppress_warnings=True,
        error_action="ignore",
        trace=True,  # 진행 상황을 보려면 True로 설정,
    )

    return model


def do_testing_arima(model: ARIMA, test_data: pd.Series):
    pred_result: pd.Series = model.predict(n_periods=len(test_data))

    mae = mean_absolute_error(test_data, pred_result)
    rmse = root_mean_squared_error(test_data, pred_result)

    return mae, rmse



def test_visualize():
    output_root_dir = "./output/arima-01-01-simple"

    with open(f"{output_root_dir}/last_model_1630008100.pkl", "rb") as f:
        last_model: BestARIMA = pickle.load(f)

    test_raw = pd.read_hdf("../datasets/metr-imc/metr-imc-test.h5")
    test_target_data = test_raw["1630008100"]

    print("Test:", test_target_data.index.min(), "to", test_target_data.index.max())

    last_pred_result: pd.Series = last_model.best_model.predict(n_periods=len(test_target_data))
    last_pred_result.index = test_target_data.index

    plt.plot(test_target_data, label="True")
    plt.plot(last_pred_result, label="Pred")

    plt.legend()

    plt.show()


# 결과는 13일이 가장 좋은 결과를 보임