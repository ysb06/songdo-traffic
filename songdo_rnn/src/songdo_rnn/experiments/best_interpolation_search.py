import glob
import logging
import os
import random
from datetime import datetime
from typing import Optional

import yaml
from lightning import Trainer
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import WandbLogger
from metr.components.metr_imc import TrafficData
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    root_mean_squared_error,
)

import wandb

from ..dataset import TrafficRatioDataModule
from ..model import SongdoTrafficLightningModel
from ..preprocessing.missing import process_missing
from ..preprocessing.outlier import process_outlier
from ..test import evaluate_model
from ..utils import fix_seed, symmetric_mean_absolute_percentage_error, non_zero_mape

OUTPUT_DIR = "./output/predictions"
TARGET_DATA_PATH = "./output/sync_processed"

logger = logging.getLogger(__name__)


def do_experiment(
    sensor_sample_rate=0.025,
    seed=42,
    target_data_path=TARGET_DATA_PATH,
    skip_preprocessing=False,
):
    fix_seed(seed)

    if not skip_preprocessing:
        process_outlier()
        process_missing()

    targets = glob.glob(f"{target_data_path}/*.h5", recursive=True)
    training_id = datetime.now().strftime("%Y%m%d%H%M%S")
    for idx_1, target in enumerate(targets):
        logger.info(f"Training model with {target}")
        raw = TrafficData.import_from_hdf(target)
        sensor_list = raw.data.columns

        sample_num = round(len(sensor_list) * sensor_sample_rate)
        target_sensors = random.sample(range(len(sensor_list)), sample_num)

        for idx_2, sensor_id in enumerate(target_sensors):
            logger.info(f"Training Model[Data {idx_1 + 1}/{len(targets)}, Sensor {idx_2 + 1}/{len(target_sensors)}]")
            train_traffic_model(
                training_id=training_id,
                traffic_data_path=target,
                target_sensor_idx=sensor_id,
            )


def train_traffic_model(
    # 1. 하이퍼 파라미터 설정
    training_id: Optional[str] = None,
    model_postfix: Optional[str] = None,
    # 1-1. 데이터 경로 및 기간 설정
    traffic_data_path: str = "./output/all_processed/abs_cap-linear.h5",
    start_datetime: Optional[str] = "2024-01-01 00:00:00",
    end_datetime: Optional[str] = "2024-08-31 23:00:00",
    target_sensor_idx: int = 0,
    # 1-2. 데이터 전처리 관련
    training_set_ratio: float = 0.8,
    validation_set_ratio: float = 0.1,
    seq_length: int = 24,
    # 1-3. 모델 관련
    input_dim: int = 1,
    hidden_dim: int = 32,
    num_layers: int = 1,
    output_dim: int = 1,
    # 1-4. 학습 관련
    epochs: int = 50,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    lr_step_size: int = 256,
    lr_gamma: float = 0.6,
    # 1-5. 기타
    verbose: bool = False,
):
    model_postfix = f"_{model_postfix}" if model_postfix is not None else ""

    # 2. 데이터 모듈 생성
    data_module = TrafficRatioDataModule(
        traffic_data_path=traffic_data_path,
        start_datetime=start_datetime,
        end_datetime=end_datetime,
        target_traffic_sensor=target_sensor_idx,
        training_set_ratio=training_set_ratio,
        validation_set_ratio=validation_set_ratio,
        seq_length=seq_length,
        batch_size=batch_size,
    )

    # 3. LightningModule 생성
    traffic_model = SongdoTrafficLightningModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        output_dim=output_dim,
        learning_rate=learning_rate,
        lr_step_size=lr_step_size,
        lr_gamma=lr_gamma,
    )

    # 4. Trainer 생성 및 학습
    # 4.1. 저장 경로 설정
    model_group_name = os.path.basename(traffic_data_path).split(".")[0]
    model_name = os.path.join(model_group_name, f"target_{target_sensor_idx:04d}")
    output_dir = os.path.join(OUTPUT_DIR, model_name)
    metrics_file_name = (
        f"metrics{model_postfix}.yaml" if model_postfix is not None else "metrics.yaml"
    )
    logger.info(f"Output Path: {output_dir}")

    # 이미 학습된 모델이 있는 경우 skip
    if os.path.exists(output_dir):
        file_list = os.listdir(output_dir)
        is_checkpoint = any([file.endswith(".ckpt") for file in file_list])
        is_metrics = any([file == metrics_file_name for file in file_list])
        if is_checkpoint and is_metrics:
            logger.info(f"Already Trained: Skipping...")
            return

    # 4.2. Wandb 연동
    run_name = f"{training_id}_{model_name}" if training_id is not None else model_name
    if model_postfix is not None:
        run_name += f"{model_postfix}"
    wandb_logger = WandbLogger(name=run_name, log_model="all", project="Songdo_RNN")

    trainer = Trainer(
        max_epochs=epochs,
        accelerator="auto",
        devices="auto",
        log_every_n_steps=1,
        default_root_dir=output_dir,
        logger=wandb_logger,
        callbacks=[
            EarlyStopping(monitor="val_loss", mode="min", patience=5, verbose=verbose),
            ModelCheckpoint(
                dirpath=output_dir,
                filename=f"best{model_postfix}" + "-{epoch:02d}-{val_loss:.2f}",
                save_top_k=1,
                monitor="val_loss",
                mode="min",
            ),
            LearningRateMonitor(logging_interval="step"),
        ],
    )

    logger.info(f"Training with {run_name}...")
    trainer.fit(traffic_model, datamodule=data_module)

    final_train_loss = trainer.callback_metrics.get("train_loss")
    final_val_loss = trainer.callback_metrics.get("val_loss")

    # 5. Predictions
    train_pred, train_true = evaluate_model(
        traffic_model,
        trainer,
        data_module.scaler,
        data_module.train_dataloader(),
    )
    test_pred, test_true = evaluate_model(
        traffic_model,
        trainer,
        data_module.scaler,
        data_module.predict_dataloader(),
    )

    # 6. Metrics 계산
    train_mae = mean_absolute_error(train_true, train_pred)
    train_rmse = root_mean_squared_error(train_true, train_pred)
    # True의 0값 때문에 MAPE 계산이 어려움
    # train_mape = mean_absolute_percentage_error(train_true, train_pred)
    train_mape, train_mape_zero_excluded = non_zero_mape(train_true, train_pred)
    train_smape = symmetric_mean_absolute_percentage_error(train_true, train_pred)

    test_mae = mean_absolute_error(test_true, test_pred)
    test_rmse = root_mean_squared_error(test_true, test_pred)
    # test_mape = mean_absolute_percentage_error(test_true, test_pred)
    test_mape, test_mape_zero_excluded = non_zero_mape(test_true, test_pred)
    test_smape = symmetric_mean_absolute_percentage_error(test_true, test_pred)

    logger.info(
        f"Train MAE: {train_mae:.6f}, RMSE: {train_rmse:.6f}, MAPE: {train_mape}, sMAPE(0-200): {train_smape:.6f}"
    )
    logger.info(
        f"Test  MAE: {test_mae:.6f}, RMSE: {test_rmse:.6f}, MAPE: {test_mape}, sMAPE(0-200): {test_smape:.6f}"
    )

    if final_train_loss is not None:
        final_train_loss = float(final_train_loss.cpu().item())
    if final_val_loss is not None:
        final_val_loss = float(final_val_loss.cpu().item())

    # -----------------------
    # 8. YAML 저장
    # -----------------------
    metrics_dict = {
        "final_train_loss": final_train_loss,
        "final_val_loss": final_val_loss,
        "train_mae": train_mae,
        "train_rmse": train_rmse,
        "train_mape": train_mape,
        "exclude_zero_in_train_mape": train_mape_zero_excluded,
        "train_smape": train_smape,
        "test_mae": test_mae,
        "test_rmse": test_rmse,
        "test_mape": test_mape,
        "exclude_zero_in_test_mape": test_mape_zero_excluded,
        "test_smape": test_smape,
    }

    yaml_path = os.path.join(output_dir, metrics_file_name)
    with open(yaml_path, "w") as f:
        yaml.safe_dump(metrics_dict, f)

    logger.info(f"Metrics saved to {yaml_path}")
    wandb.finish()


if __name__ == "__main__":
    train_traffic_model()
