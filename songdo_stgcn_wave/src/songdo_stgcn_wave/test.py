import os
import random
from dataclasses import asdict
from datetime import datetime
import logging

import dgl
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

import wandb
from stgcn_wave.load_data import data_transform
from stgcn_wave.model import STGCN_WAVE
from stgcn_wave.sensors2graph import get_adjacency_matrix
from stgcn_wave.utils import evaluate_metric, evaluate_model
from wandb import Config

from .utils import HyperParams, get_auto_device, fix_seed
from metr.components.adj_mx import import_adj_mx
from metr.dataloader import MetrDataset

logger = logging.getLogger(__name__)


def test_model(config: HyperParams):
    logger.info(f"Test for {config.dataset_name}")
    training_device = get_auto_device()
    fix_seed(config.seed)
    adj_mx_raw = import_adj_mx(config.adj_mx_filepath)
    sparse_mx = sp.coo_matrix(adj_mx_raw.adj_mx)
    G = dgl.from_scipy(sparse_mx)

    dataset = MetrDataset.from_file(
        config.tsfilepath,
        config.window,
        config.pred_len,
        config.missing_labels_filepath,
    )
    test_loader = DataLoader(
        dataset,
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
        training_device,  # Device
        config.control_str,
    )
    best_model.load_state_dict(torch.load(config.savemodelpath))
    MAE, MAPE, RMSE = evaluate_model_(
        test_loader, dataset.scaler_for_all, best_model, training_device
    )
    test_result = {"test_MAE": MAE, "test_RMSE": RMSE, "test_MAPE": MAPE}
    logger.info(f"Test Result:\r\n{test_result}")


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
