import os
from lightning import Trainer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from .dataset import TrafficRatioDataModule
from .model import SongdoTrafficLightningModel


def train_model(
    traffic_data_path: str,
    start_datetime: str,
    end_datetime: str,
    target_traffic_sensor: int = 0,
    training_data_ratio: float = 0.7,
    validation_data_ratio: float = 0.1,
    seq_length: int = 24,
    input_dim: int = 1,
    hidden_dim: int = 32,
    num_layers: int = 1,
    output_dim: int = 1,
    epochs: int = 50,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    save_path: str = None,
):
    # 2. 데이터 모듈 생성
    data_module = TrafficRatioDataModule(
        traffic_data_path=traffic_data_path,
        start_datetime=start_datetime,
        end_datetime=end_datetime,
        target_traffic_sensor=target_traffic_sensor,
        training_set_ratio=training_data_ratio,
        validation_set_ratio=validation_data_ratio,
        seq_length=seq_length,
        batch_size=batch_size,
    )

    # 3. LightningModule 생성
    model = SongdoTrafficLightningModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        output_dim=output_dim,
        learning_rate=learning_rate,
    )

    # 4. Trainer 생성 및 학습
    trainer = Trainer(
        max_epochs=epochs,
        accelerator="auto",
        devices="auto",
        log_every_n_steps=1,
        callbacks=[EarlyStopping(monitor="val_loss", patience=5)],
        default_root_dir=save_path,
    )

    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    # 1. 하이퍼 파라미터 설정
    # 1-1. 데이터 경로 및 기간 설정
    traffic_data_path = "../datasets/metr-imc/metr-imc.h5"
    start_datetime = None
    end_datetime = "2024-08-31 23:00:00"

    train_model(
        traffic_data_path=traffic_data_path,
        start_datetime=start_datetime,
        end_datetime=end_datetime,
    )
