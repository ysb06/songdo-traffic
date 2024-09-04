import logging
import os

import dgl
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd

from stgcn_wave.model import STGCN_WAVE

from .utils import HyperParams, get_auto_device, fix_seed
from metr.components.adj_mx import import_adj_mx
from metr.dataloader import MetrDataset

logger = logging.getLogger(__name__)


def test_model(config: HyperParams):
    logger.info(f"Test for {config.dataset_name}")
    test_device = get_auto_device()
    fix_seed(config.seed)
    adj_mx_raw = import_adj_mx(config.adj_mx_filepath)
    sparse_mx = sp.coo_matrix(adj_mx_raw.adj_mx)
    G = dgl.from_scipy(sparse_mx)
    G = G.to(test_device)

    dataset = MetrDataset.from_file(
        config.tsfilepath,
        config.window,
        config.pred_len,
        config.missing_labels_filepath,
    )
    _, _, test_dataset, scaler = dataset.split()
    start_index = test_dataset.indices[0]
    end_index = test_dataset.indices[-1]
    test_size = end_index - start_index + 1

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        collate_fn=MetrDataset.collate_fn_with_missing,
    )
    best_model = STGCN_WAVE(
        config.channels,  # Blocks
        config.window,  # History Length
        dataset.num_nodes,  # Number of Roads(Routes)
        G,
        config.drop_rate,  # Dropout Rate (Normally 0)
        config.num_layers,
        test_device,  # Device
        config.control_str,
    )
    best_model.load_state_dict(torch.load(config.savemodelpath, map_location=test_device))
    MAE, MAPE, RMSE = evaluate_model_(
        test_loader, scaler, best_model, test_device
    )
    test_result = {"test_MAE": MAE, "test_RMSE": RMSE, "test_MAPE": MAPE}
    logger.info(f"Test Result:\r\n{test_result}")

    y_true, y_hat = predict(test_loader, scaler, best_model, test_device)
    time_index = dataset.raw_df.index[-test_size:]
    idx_to_sensor_id = {v: k for k, v in adj_mx_raw.sensor_id_to_idx.items()}
    columns = [idx_to_sensor_id[i] for i in range(y_true.shape[1])]

    y_true_df = pd.DataFrame(y_true, index=time_index, columns=columns)
    y_hat_df = pd.DataFrame(y_hat, index=time_index, columns=columns)
    
    output_dir = "./output/"
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, "predictions.xlsx")
    logger.info(f"Saving predictions to {output_path}")
    with pd.ExcelWriter(output_path) as writer:
        y_true_df.to_excel(writer, sheet_name="Y_True")
        y_hat_df.to_excel(writer, sheet_name="Y_Hat")


def predict(
    dataloader: DataLoader,
    scaler: StandardScaler,
    model: nn.Module,
    device: torch.device,
):
    model = model.to(device)
    model.eval()

    y_true_list = []
    y_hat_list = []
    with torch.no_grad():
        for x, y, _ in tqdm(dataloader):
            x: Tensor = x.to(device)
            y: Tensor = y.to(device)
            y_pred: Tensor = model(x)
            y_pred = y_pred.view(len(x), -1)

            y_true = scaler.inverse_transform(y.cpu().numpy()).squeeze()
            y_hat = scaler.inverse_transform(y_pred.cpu().numpy()).squeeze()

            y_true_list.append(y_true)
            y_hat_list.append(y_hat)
    
    y_true_series = np.concatenate(y_true_list, axis=0)
    y_hat_series = np.concatenate(y_hat_list, axis=0)

    return y_true_series, y_hat_series

def evaluate_model_(
    dataloader: DataLoader,
    scaler: StandardScaler,
    model: nn.Module,
    device: torch.device,
):
    model = model.to(device)
    model.eval()

    mae_sum, mape_sum, mse_sum = 0.0, 0.0, 0.0
    valid_count = 0  # 유효한 데이터 포인트의 수를 셀 변수
    with torch.no_grad():
        for x, y, mis_y in tqdm(dataloader):
            x: Tensor = x.to(device)
            y: Tensor = y.to(device)
            y_pred: Tensor = model(x)
            y_pred = y_pred.view(len(x), -1)

            y_true = scaler.inverse_transform(y.cpu().numpy()).squeeze()
            y_hat = scaler.inverse_transform(y_pred.cpu().numpy()).squeeze()

            mask: Tensor = ~mis_y
            y_true_valid = y_true[mask]
            y_hat_valid = y_hat[mask]

            d_error: np.ndarray = np.abs(y_true_valid - y_hat_valid)
            mae_sum += d_error.sum()
            mape_sum += np.where(y_true_valid != 0, (d_error / y_true_valid), 0).sum()
            mse_sum += (d_error**2).sum()

            valid_count += mask.sum()

    MAE = mae_sum / valid_count
    MAPE = mape_sum / valid_count
    RMSE = np.sqrt(mse_sum / valid_count)

    return MAE, MAPE, RMSE
