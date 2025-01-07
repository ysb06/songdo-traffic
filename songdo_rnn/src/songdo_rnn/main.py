import logging

from lightning import Trainer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    root_mean_squared_error,
)

from .dataset import TrafficDataModule
from .model import SongdoTrafficLightning
from .test import evaluate_model
from .utils import symmetric_mean_absolute_percentage_error

logger = logging.getLogger(__name__)

# 1. 하이퍼 파라미터 설정
# 1-1. 데이터 경로 및 기간 설정
traffic_data_path = "../datasets/metr-imc/metr-imc.h5"
# start_datetime = "2024-03-01 00:00:00"
# end_datetime = "2024-09-30 23:00:00"
start_datetime = None
end_datetime = "2024-08-31 23:00:00"
target_traffic_sensor = 0

# 1-2. 데이터 전처리 관련
training_data_ratio = 0.8
seq_length = 24

# 1-3. 모델 관련
input_dim = 1
hidden_dim = 32
num_layers = 1
output_dim = 1

# 1-4. 학습 관련
epochs = 50
batch_size = 64
learning_rate = 0.001

if __name__ == "__main__":
    # 2. 데이터 모듈 생성
    data_module = TrafficDataModule(
        traffic_data_path=traffic_data_path,
        start_datetime=start_datetime,
        end_datetime=end_datetime,
        target_traffic_sensor=target_traffic_sensor,
        training_data_ratio=training_data_ratio,
        seq_length=seq_length,
        batch_size=batch_size,
    )

    # 3. LightningModule 생성
    model = SongdoTrafficLightning(
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
        callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=5)],
    )

    trainer.fit(model, datamodule=data_module)

    # 5. Predictions
    train_pred, train_true = evaluate_model(model, trainer, data_module)
    test_pred, test_true = evaluate_model(model, trainer, data_module)

    # 6. Metrics 계산
    train_mae = mean_absolute_error(train_true, train_pred)
    train_rmse = root_mean_squared_error(train_true, train_pred)
    # True의 0값 때문에 MAPE 계산이 어려움
    # train_mape = mean_absolute_percentage_error(train_true, train_pred)
    train_smape = symmetric_mean_absolute_percentage_error(train_true, train_pred)

    test_mae = mean_absolute_error(test_true, test_pred)
    test_rmse = root_mean_squared_error(test_true, test_pred)
    # test_mape = mean_absolute_percentage_error(test_true, test_pred)
    test_smape = symmetric_mean_absolute_percentage_error(test_true, test_pred)

    logger.info(
        f"Train MAE: {train_mae:.6f}, RMSE: {train_rmse:.6f}, sMAPE(0-200): {train_smape:.6f}"
    )
    logger.info(
        f"Test MAE : {test_mae:.6f}, RMSE: {test_rmse:.6f}, sMAPE(0-200): {test_smape:.6f}"
    )
