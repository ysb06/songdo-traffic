import logging
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from metr.components import TrafficData
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, mean_absolute_percentage_error

from .dataset import TrafficDataset, split_train_test
from .model import TrafficRNN, TrafficLSTM
from .utils import get_auto_device

logger = logging.getLogger(__name__)


# ==========================
# 0. Hyperparameters
# ==========================
traffic_data_path = "../datasets/metr-imc/metr-imc.h5"
start_datetime = "2024-03-01 00:00:00"
end_datetime = "2024-09-30 23:00:00"
target_traffic_sensor = 0

# Dataset-related
training_data_ratio = 0.8
seq_length = 24

# Training-related
epochs = 50
batch_size = 64
learning_rate = 0.001

# ==========================
# 1. 데이터 불러오기
# ==========================
traffic_data = TrafficData.import_from_hdf(traffic_data_path)
logger.info(
    f"Traffic Data Loaded: From {traffic_data.start_time} to {traffic_data.end_time}"
)

if start_datetime is not None:
    traffic_data.start_time = start_datetime
if end_datetime is not None:
    traffic_data.end_time = end_datetime
logger.info(
    f"Using Traffic Data: From {traffic_data.start_time} to {traffic_data.end_time}"
)

# target_traffic_sensor 번째 센서 데이터만 사용
data = traffic_data.data.iloc[:, target_traffic_sensor]

# ==========================
# 2. 커스텀 Dataset 생성
#    (데이터 전처리, 시퀀스화는 TrafficDataset 내부에서 처리)
# ==========================
device = get_auto_device()
logger.info(f"Device: {device}")

dataset = TrafficDataset(data=data.values, seq_length=seq_length)

# ==========================
# 3. Train/Test Split
# ==========================
train_subset, test_subset = split_train_test(dataset, train_ratio=training_data_ratio)

logger.info(f"Train subset size: {len(train_subset)}")
logger.info(f"Test subset size : {len(test_subset)}")

train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

# ==========================
# 4. 모델 정의
# ==========================
model = TrafficRNN(
    input_dim=1, hidden_dim=32, num_layers=1, output_dim=1  # 단일 센서
).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# ==========================
# 5. 모델 학습
# ==========================
for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0

    for x_batch, y_batch in tqdm(
        train_loader, total=len(train_loader), desc=f"Epoch {epoch+1}"
    ):
        # x_batch.shape: (seq_length, batch_size, 1) or (batch_size, seq_length, 1)
        x_batch: torch.Tensor = x_batch.to(device)
        y_batch: torch.Tensor = y_batch.to(device)

        optimizer.zero_grad()
        outputs: torch.Tensor = model(x_batch)  # 예측값
        loss: torch.Tensor = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        # 통계
        epoch_loss += loss.item() * x_batch.size(0)

    # 평균 Loss
    epoch_loss /= len(train_loader.dataset)

    if (epoch + 1) % 10 == 0:
        logger.info(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.6f}")

# ==========================
# 6. 예측 및 결과 확인
# ==========================
model.eval()

# - train_loader, test_loader 순회하며 예측값/실제값 모으기
train_preds_list = []
train_true_list = []
with torch.no_grad():
    for x_batch, y_batch in train_loader:
        x_batch: torch.Tensor = x_batch.to(device)
        y_batch: torch.Tensor = y_batch.to(device)

        preds: torch.Tensor = model(x_batch)
        train_preds_list.append(preds.cpu().numpy())
        train_true_list.append(y_batch.cpu().numpy())

test_preds_list = []
test_true_list = []
with torch.no_grad():
    for x_batch, y_batch in test_loader:
        x_batch: torch.Tensor = x_batch.to(device)
        y_batch: torch.Tensor = y_batch.to(device)

        preds: torch.Tensor = model(x_batch)
        test_preds_list.append(preds.cpu().numpy())
        test_true_list.append(y_batch.cpu().numpy())

# 리스트를 하나로 이어붙임
train_preds_arr = np.concatenate(train_preds_list, axis=0)
train_true_arr = np.concatenate(train_true_list, axis=0)
test_preds_arr = np.concatenate(test_preds_list, axis=0)
test_true_arr = np.concatenate(test_true_list, axis=0)

# ==========================
# 7. 정규화 해제(inverse_transform)
#    - Dataset 내부에서 MinMaxScaler를 활용했으므로,
#      dataset.scaler.inverse_transform()를 사용
# ==========================
scaler = dataset.scaler
train_preds_inv = scaler.inverse_transform(train_preds_arr)
train_y_inv = scaler.inverse_transform(train_true_arr)
test_preds_inv = scaler.inverse_transform(test_preds_arr)
test_y_inv = scaler.inverse_transform(test_true_arr)

# ==========================
# 8. 결과 DataFrame 구성
# ==========================

train_start_idx = seq_length
train_end_idx = train_start_idx + len(train_preds_inv)
test_start_idx = train_end_idx - 24
test_end_idx = train_end_idx + len(test_preds_inv)

train_pred_result_df = pd.DataFrame(
    {
        "Train True": train_y_inv.squeeze(1),
        "Train Predicted": train_preds_inv.squeeze(1),
    },
    index=data.index[train_start_idx:train_end_idx],
)

test_pred_result_df = pd.DataFrame(
    {"Test True": test_y_inv.squeeze(1), "Test Predicted": test_preds_inv.squeeze(1)},
    index=data.index[test_start_idx:test_end_idx],
)
# ==========================
# 9. Metrics 계산
# ==========================

train_mae = mean_absolute_error(train_y_inv, train_preds_inv)
train_rmse = root_mean_squared_error(train_y_inv, train_preds_inv)
train_mape = mean_absolute_percentage_error(train_y_inv, train_preds_inv)

test_mae = mean_absolute_error(test_y_inv, test_preds_inv)
test_rmse = root_mean_squared_error(test_y_inv, test_preds_inv)
test_mape = mean_absolute_percentage_error(test_y_inv, test_preds_inv)

logger.info(f"Train MAE: {train_mae:.6f}, RMSE: {train_rmse:.6f}")
logger.info(f"Test MAE : {test_mae:.6f}, RMSE: {test_rmse:.6f}")

# ==========================
# 10. 시각화
# ==========================
plt.figure(figsize=(10, 6))
sns.lineplot(train_pred_result_df)
plt.title("Train Results")
plt.xlabel("Datetime")
plt.ylabel("Traffic Volume")
plt.legend()
plt.show()

sns.lineplot(test_pred_result_df)
plt.title("Test Reuslts")
plt.xlabel("Datetime")
plt.ylabel("Traffic Volume")
plt.legend()
plt.show()
