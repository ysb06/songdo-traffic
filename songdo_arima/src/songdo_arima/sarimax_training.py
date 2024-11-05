import os
import pickle

import pandas as pd
from metr.components import TrafficData
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error

from songdo_arima.utils import HyperParams


def train(config: HyperParams):
    output_dir = config.output_root_dir
    model_output_dir = os.path.join(output_dir, "models")
    result_filepath = os.path.join(output_dir, "results.pkl")

    os.makedirs(model_output_dir, exist_ok=True)

    raw = TrafficData.import_from_hdf(config.traffic_training_data_path)
    raw_data = raw.data
    data_size = raw_data.shape[0]
    train_size = int(data_size * config.training_data_ratio)
    train_data = raw_data.iloc[:train_size, :]
    valid_data = raw_data.iloc[train_size:, :]

    target_sensor = raw_data.columns[0]
    train_target = train_data[target_sensor]
    valid_target = valid_data[target_sensor]

    train_sensor(train_target, valid_target, model_output_dir)

def train_sensor(train_data: pd.Series, valid_data: pd.Series, model_output_dir: str):
    # auto_arima 모델 학습
    model = auto_arima(
        train_data,
        start_p=0, max_p=3,
        start_q=0, max_q=2,
        d=None,            # 'd' 값을 자동으로 결정
        start_P=0, max_P=1,
        start_Q=0, max_Q=1,
        D=None,            # 'D' 값을 자동으로 결정
        m=24,              # 계절 주기 (시간별 데이터의 일일 계절성)
        seasonal=True,
        stepwise=True,     # 스텝와이즈 알고리즘 사용으로 계산 시간 단축
        suppress_warnings=True,
        error_action='ignore',
        trace=True        # 진행 상황을 보려면 True로 설정
    )

    # 검증 데이터에 대한 예측
    n_periods = len(valid_data)
    forecast = model.predict(n_periods=n_periods)
    forecast = pd.Series(forecast, index=valid_data.index)

    # MAE 계산
    mae = mean_absolute_error(valid_data, forecast)
    print(f'Auto ARIMA 모델 MAE: {mae}')

    # 모델 저장
    model_output_path = os.path.join(model_output_dir, "best_model.pkl")
    with open(model_output_path, 'wb') as pkl:
        pickle.dump(model, pkl)
