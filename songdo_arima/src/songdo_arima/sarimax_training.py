import logging
import os
import random
from typing import Dict, List, Optional, Tuple

import pandas as pd
import yaml
from joblib import Parallel, delayed
from metr.components import TrafficData
from pmdarima import ARIMA, auto_arima
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from tqdm import tqdm

logger = logging.getLogger(__name__)

training_timestamp = pd.Timestamp.now().strftime("%d %b %Y, %H:%M:%S")


def train_model(traffic_data_path: str, config: Dict):
    random.seed = config["random_seed"]
    logger.info(f"Random Seeded: {random.seed}")

    raw = TrafficData.import_from_hdf(traffic_data_path)
    raw_dir = os.path.dirname(traffic_data_path)
    output_dir = os.path.join(raw_dir, "sarimax")
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    start_datetime = config["start_datetime"]
    end_datetime = config["end_datetime"]
    if start_datetime is not None:
        logger.info(f"Start datetime: {start_datetime}")
        raw.start_time = start_datetime
    if end_datetime is not None:
        logger.info(f"End datetime: {end_datetime}")
        raw.end_time = end_datetime

    raw_data = raw.data
    data_size = raw_data.shape[0]
    train_size = int(data_size * config["training_data_ratio"])
    train_data = raw_data.iloc[:train_size, :]
    valid_data = raw_data.iloc[train_size:, :]

    target_columns = raw_data.columns.to_list()
    target_sample_rate: Optional[float] = config["target_sample_rate"]
    if target_sample_rate is not None:
        total_columns = len(target_columns)
        num_columns_to_select = max(1, int(total_columns * target_sample_rate))
        rnd = random.Random(config["random_seed"])
        target_columns = rnd.sample(target_columns, num_columns_to_select)
        logger.info(
            f"Sample Columns Selected: {len(raw_data.columns)} -> {len(target_columns)}"
        )
        if len(target_columns) < 10:
            logger.info(f"Sample Columns:\r\n{target_columns}")

    ## ------- Using Parallel
    results = Parallel(n_jobs=3)(
        delayed(train_sensor)(
            train_data[column], valid_data[column], output_dir, config
        )
        for column in tqdm(target_columns)
    )

    ## ------- Not using Parallel
    # results: List[Tuple[float, float]] = []
    # with tqdm(total=len(target_columns)) as pbar:
    #     for column in target_columns:
    #         pbar.set_description(f"Training {column}")
    #         pbar.refresh()
    #         result = train_sensor(
    #             train_data[column], valid_data[column], output_dir, config
    #         )
    #         results.append(result)
    #         pbar.update(1)

    # MAE와 RMSE의 합계 및 개수 계산
    mae_list = [mae for mae, _ in results if mae is not None]
    rmse_list = [rmse for _, rmse in results if rmse is not None]

    if len(mae_list) != len(results):
        logger.warning(f"Some MAE values are missing: {len(results) - len(mae_list)}")
    
    if len(rmse_list) != len(results):
        logger.warning(f"Some RMSE values are missing: {len(results) - len(rmse_list)}")

    mae_sum = sum(mae_list)
    rmse_sum = sum(rmse_list)
    mae_count = len(mae_list)
    rmse_count = len(rmse_list)

    config_output_path = os.path.join(output_dir, "results.yaml")

    result = {"timestamp": training_timestamp}
    result.update(config)
    result["data_shape"] = [raw_data.shape[0], raw_data.shape[1]]
    result["training_sampled_columns_size"] = len(target_columns)
    result["train_size"] = train_size
    result["valid_size"] = data_size - train_size
    result["mean_MAE"] = mae_sum / mae_count
    result["mean_RMSE"] = rmse_sum / rmse_count

    with open(config_output_path, "w") as f:
        yaml.dump(result, f)


def train_sensor(
    train_data: pd.Series,
    valid_data: pd.Series,
    model_output_dir: str,
    config: Dict,
    force_training: bool = False,
) -> Tuple[float, float]:
    model_output_path = os.path.join(
        model_output_dir, "results", f"{train_data.name}.yaml"
    )
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)

    is_stationary_test_passed = True
    if not os.path.exists(model_output_path) or force_training:
        try:
            model: ARIMA = auto_arima(
                train_data,
                **config["hyperparams"],
                seasonal=True,
                stepwise=True,
                error_action="ignore",
            )
        except ValueError as e:
            logger.error(f"ValueError: {e}")
            is_stationary_test_passed = False
            model: ARIMA = auto_arima(
                train_data,
                **config["hyperparams"],
                seasonal=True,
                stepwise=True,
                stationary=True,
            )

        # 검증 데이터에 대한 예측
        n_periods = len(valid_data)
        is_prediction_failed_error = False
        mae = None
        rmse = None

        try:
            forecast = model.predict(n_periods=n_periods)
            forecast = pd.Series(forecast, index=valid_data.index)

            # MAE 및 RMSE 계산
            try:
                mae = mean_absolute_error(valid_data, forecast).item()
            except ValueError as e:
                logger.error(f"MAE not calculated: {e}")
                # Todo: Interpolation이 제대로 되지 않았을 때 어떻게 처리할지 고민 필요
            try:
                rmse = root_mean_squared_error(valid_data, forecast).item()
            except ValueError as e:
                logger.error(f"RMSE not calculated: {e}")
        except ValueError as e:
            logger.error(f"Prediction Failed: {e}")
            is_prediction_failed_error = True
            forecast = None    

        result = {
            "training_timestamp": training_timestamp,
            "recording_timestamp": pd.Timestamp.now().strftime("%d %b %Y, %H:%M:%S"),
        }
        result["name"] = train_data.name
        result["period_for_training"] = {
            "start": str(train_data.index[0]),
            "end": str(train_data.index[-1]),
        }
        result["period_for_validation"] = {
            "start": str(valid_data.index[0]),
            "end": str(valid_data.index[-1]),
        }
        best_params = model.get_params()
        result["best_params"] = {
            "order": list(best_params["order"]),
            "seasonal_order": list(best_params["seasonal_order"]),
            "maxiter": best_params["maxiter"],
            "scoring": best_params["scoring"],
            "with_intercept": best_params["with_intercept"],
        }
        result["mae"] = mae
        result["rmse"] = rmse
        result["stationary_test_passed"] = is_stationary_test_passed
        result["prediction_failed_error"] = is_prediction_failed_error

        # 모델 저장
        with open(model_output_path, "w") as f:
            yaml.dump(result, f)
    else:
        with open(model_output_path, "r") as f:
            result: Dict = yaml.load(f, Loader=yaml.FullLoader)
            mae = result["mae"]
            rmse = result["rmse"]

    return mae, rmse
