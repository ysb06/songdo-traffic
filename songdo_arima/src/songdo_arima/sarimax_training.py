import os
import pickle
from dataclasses import asdict

import pandas as pd
import yaml
# 병렬 처리를 위한 라이브러리 임포트
from joblib import Parallel, delayed
from metr.components import TrafficData
from pmdarima import ARIMA, auto_arima
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from songdo_arima.utils import HyperParams
from tqdm import tqdm


class BestARIMA:
    def __init__(self, model: ARIMA) -> None:
        self.best_model = model
        self.result = {}

def train(config: HyperParams) -> dict:
    output_dir = config.output_root_dir
    model_output_dir = os.path.join(output_dir, "models")
    os.makedirs(model_output_dir, exist_ok=True)

    raw = TrafficData.import_from_hdf(config.traffic_training_data_path)
    raw_data = raw.data
    data_size = raw_data.shape[0]
    train_size = int(data_size * config.training_data_ratio)
    train_data = raw_data.iloc[:train_size, :]
    valid_data = raw_data.iloc[train_size:, :]

    # 병렬 연산을 통해 각 센서에 대한 모델 학습
    results = Parallel(n_jobs=2)(
        delayed(train_sensor)(train_data[column], valid_data[column], model_output_dir)
        for column in tqdm(raw_data.columns)
    )

    # MAE와 RMSE의 합계 및 개수 계산
    mae_sum = sum(mae for mae, _ in results)
    rmse_sum = sum(rmse for _, rmse in results)
    mae_count = len(results)
    rmse_count = len(results)

    config_output_path = os.path.join(model_output_dir, "results.yaml")
    config_dict = asdict(config)
    config_dict["data_shape"] = raw_data.shape
    config_dict["train_size"] = train_size
    config_dict["valid_size"] = data_size - train_size
    config_dict["mean_MAE"] = mae_sum / mae_count
    config_dict["mean_RMSE"] = rmse_sum / rmse_count

    with open(config_output_path, "w") as f:
        yaml.dump(config_dict, f)

    return config_dict


def train_sensor(train_data: pd.Series, valid_data: pd.Series, model_output_dir: str):
    model_output_path = os.path.join(model_output_dir, f"{train_data.name}.pkl")

    if not os.path.exists(model_output_path):
        # auto_arima 모델 학습
        model: ARIMA = auto_arima(
            train_data,
            start_p=0,
            max_p=3,
            start_q=0,
            max_q=3,
            d=None,  # 'd' 값을 자동으로 결정
            start_P=0,
            max_P=1,
            start_Q=0,
            max_Q=1,
            D=None,  # 'D' 값을 자동으로 결정
            m=24,  # 계절 주기 (시간별 데이터의 일일 계절성)
            seasonal=True,
            stepwise=True,  # 스텝와이즈 알고리즘 사용으로 계산 시간 단축
            suppress_warnings=True,
            error_action="ignore",
            trace=False,  # 진행 상황을 보려면 True로 설정
        )

        # 검증 데이터에 대한 예측
        n_periods = len(valid_data)
        forecast = model.predict(n_periods=n_periods)
        forecast = pd.Series(forecast, index=valid_data.index)

        # MAE 및 RMSE 계산
        mae = mean_absolute_error(valid_data, forecast)
        rmse = root_mean_squared_error(valid_data, forecast)

        best_result = BestARIMA(model)
        best_result.result["MAE"] = mae
        best_result.result["RMSE"] = rmse

        # 모델 저장
        with open(model_output_path, "wb") as pkl:
            pickle.dump(best_result, pkl)
            order = model.order
            seasonal_order = model.seasonal_order
            print(f"센서 {train_data.name} is trained: {order} {seasonal_order}")
    else:
        with open(model_output_path, "rb") as pkl:
            best_result: BestARIMA = pickle.load(pkl)
            mae = best_result.result["MAE"]
            rmse = best_result.result["RMSE"]
            order = best_result.best_model.order
            seasonal_order = best_result.best_model.seasonal_order
            print(f"센서 {train_data.name} is already trained: {order} {seasonal_order}")

    return mae, rmse
