import logging
import os
import random
from collections import defaultdict
from glob import glob
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from lightning import Trainer
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import WandbLogger
from metr.components import TrafficData
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from tqdm import tqdm

from .lightning.dataset import DataInitializationError, TrafficDataModule
from .lightning.model.rnn import TrafficVolumePredictionModule
from .utils import fix_seed, symmetric_mean_absolute_percentage_error

logger = logging.getLogger(__name__)

# Hyperparameters
seq_length: int = 24
input_dim: int = 1
hidden_dim: int = 32
num_layers: int = 1
output_dim: int = 1
epochs: int = 50
batch_size: int = 64

learning_rate: float = 0.001
lr_step_size: int = 100
lr_gamma: float = 0.1

k = 30  # Number of sensors to select
fix_seed(42)  # Set random seed for reproducibility


def get_output_dir(output_dir: str, name: str, sensor_id: str) -> str:
    return os.path.join(output_dir, name, sensor_id)


def do_prediction_test(
    test_set: List[Tuple[pd.DataFrame, str]], true_df: pd.DataFrame, output_dir: str
):
    for target_df, target_name in test_set:
        target_sensors = random.sample(
            list(target_df.columns), min(k, len(target_df.columns))
        )

        for sensor_name in target_sensors:
            sensor_data = pd.DataFrame({sensor_name: target_df[sensor_name]})
            test_data = pd.DataFrame({sensor_name: true_df[sensor_name]})

            sensor_output_dir = get_output_dir(output_dir, target_name, sensor_name)
            if (
                os.path.exists(sensor_output_dir)
                and len(os.listdir(sensor_output_dir)) > 0
            ):
                print(f"Skip: {sensor_output_dir} already exists")
                continue  # 이미 학습된 모델이 있는 경우 skip
            os.makedirs(sensor_output_dir, exist_ok=True)

            if sensor_data.empty or test_data.empty:
                logger.error(
                    f"\r\n{'-'*30}\r\nSkipping sensor {sensor_name} in {target_name} due to empty data.\r\n{'-'*30}"
                )
                continue
            try:
                data_module, model = train_model(
                    sensor_data, test_data, sensor_output_dir
                )
            except DataInitializationError as e:
                logger.error(
                    f"\r\n{'-'*30}\r\nSkipping sensor {sensor_name} in {target_name} due to data initialization error: {e}\r\n{'-'*30}"
                )
                continue
            # 예측 테스트
            test_true, test_pred = predict_by_model(data_module, sensor_output_dir)
            test_mae = mean_absolute_error(test_true, test_pred)
            test_rmse = root_mean_squared_error(test_true, test_pred)
            test_smape = symmetric_mean_absolute_percentage_error(test_true, test_pred)

            metrics_dict = {
                "MAE": test_mae,
                "RMSE": test_rmse,
                "sMAPE": test_smape,
            }

            metrics_file_name = f"metrics_{sensor_name}.yaml"
            yaml_path = os.path.join(sensor_output_dir, metrics_file_name)
            with open(yaml_path, "w") as f:
                yaml.safe_dump(metrics_dict, f)
                logger.info(f"Metrics saved to {yaml_path}")

            # 메모리 해제
            del data_module, model, test_true, test_pred
            torch.cuda.empty_cache()


def aggregate_metrics(output_dir: str):
    # 메트릭 유형별로 별도의 딕셔너리 초기화
    metrics_dict = {"MAE": {}, "RMSE": {}, "sMAPE": {}}

    pred_dirs = glob(os.path.join(output_dir, "*"))
    for pred_dir in pred_dirs:
        if not os.path.isdir(pred_dir):
            continue
        model_name = os.path.basename(pred_dir).split(".")[0]

        # 각 메트릭 유형에 대해 모델별 딕셔너리 초기화
        for metric_type in metrics_dict.keys():
            metrics_dict[metric_type][model_name] = {}

        metric_paths = glob(os.path.join(pred_dir, "*", "metrics_*.yaml"))
        for metric_path in metric_paths:
            sensor_name = os.path.basename(os.path.dirname(metric_path))

            with open(metric_path, "r") as f:
                metric = yaml.safe_load(f)

                # 각 메트릭 유형마다 값 저장
                for metric_type in metrics_dict.keys():
                    metrics_dict[metric_type][model_name][sensor_name] = metric[
                        metric_type
                    ]

    # 각 메트릭 유형별로 DataFrame 생성 (인덱스가 model_name, 컬럼이 sensor_name)
    mae_df = pd.DataFrame(metrics_dict["MAE"]).T
    rmse_df = pd.DataFrame(metrics_dict["RMSE"]).T
    smape_df = pd.DataFrame(metrics_dict["sMAPE"]).T

    # 결과 확인
    print("MAE DataFrame 형태:", mae_df.shape)
    print("RMSE DataFrame 형태:", rmse_df.shape)
    print("sMAPE DataFrame 형태:", smape_df.shape)

    mae_df.to_excel(os.path.join(output_dir, "ptest_mae.xlsx"))
    rmse_df.to_excel(os.path.join(output_dir, "ptest_rmse.xlsx"))
    smape_df.to_excel(os.path.join(output_dir, "ptest_smape.xlsx"))


def train_model(
    training_data: pd.DataFrame,
    test_data: pd.DataFrame,
    output_dir: str,
):
    data_module = TrafficDataModule(
        training_df=training_data,
        test_df=test_data,
        seq_length=seq_length,
        batch_size=batch_size,
        valid_split_datetime="2024-06-01 00:00:00",
        strict_scaling=True,  # 임시 해결책. 학습 데이터가 많지 않아 가능한 해법.
    )

    traffic_model = TrafficVolumePredictionModule(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        output_dim=output_dim,
        learning_rate=learning_rate,
        lr_step_size=lr_step_size,
        lr_gamma=lr_gamma,
    )
    wandb_logger = WandbLogger(project="Songdo_LSTM", log_model="all")

    trainer = Trainer(
        max_epochs=epochs,
        accelerator="auto",
        devices="auto",
        log_every_n_steps=1,
        default_root_dir=output_dir,
        logger=wandb_logger,
        callbacks=[
            EarlyStopping(monitor="val_loss", mode="min", patience=5),
            ModelCheckpoint(
                dirpath=output_dir,
                filename="best-{epoch:02d}-{val_loss:.2f}",
                save_top_k=1,
                monitor="val_loss",
                mode="min",
            ),
            LearningRateMonitor(logging_interval="step"),
        ],
    )

    trainer.fit(traffic_model, data_module)

    return data_module, traffic_model


def predict_by_model(
    data: TrafficDataModule,
    model_dir: str,
):
    checkpoint_files = glob(os.path.join(model_dir, "best*.ckpt"))
    checkpoint_path = checkpoint_files[0]
    print(f"Using checkpoint: {os.path.basename(checkpoint_path)}")

    traffic_model = TrafficVolumePredictionModule.load_from_checkpoint(
        checkpoint_path=checkpoint_path
    )
    traffic_model.eval()
    data_loader = data.predict_dataloader()

    # -------------------
    # Prediction
    # -------------------
    test_scaled_true: List[np.ndarray] = []
    test_scaled_pred: List[np.ndarray] = []
    for idx, item in tqdm(enumerate(data_loader), total=len(data_loader)):
        x: torch.Tensor = item[0]
        y: torch.Tensor = item[1]

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
                f"[{idx + 1}/{len(data_loader)}] Batch Filtered: {x.size(0)} -> {x_filtered.size(0)}"
            )
        if x_filtered.size(0) == 0:
            print(f"[{idx + 1}/{len(data_loader)}] Batch Passed")
            continue

        x_filtered = x_filtered.to(traffic_model.device)
        y_filtered = y_filtered.to(traffic_model.device)

        y_hat: torch.Tensor = traffic_model(x_filtered)

        test_scaled_true.append(y_filtered.cpu().detach().numpy())
        test_scaled_pred.append(y_hat.cpu().detach().numpy())

    if len(test_scaled_true) == 0 or len(test_scaled_pred) == 0:
        print(f"No valid data for sensor")
        return

    test_scaled_true_arr = np.concatenate(test_scaled_true, axis=0)
    test_scaled_pred_arr = np.concatenate(test_scaled_pred, axis=0)
    scaler = data.scaler
    test_true_arr = scaler.inverse_transform(test_scaled_true_arr)
    test_pred_arr = scaler.inverse_transform(test_scaled_pred_arr)
    test_true = test_true_arr.squeeze(1)
    test_pred = test_pred_arr.squeeze(1)

    return test_true, test_pred
