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
from .test import evaluate_model_

logger = logging.getLogger(__name__)


def train_new(config: HyperParams):
    run_name = (
        f"{config.dataset_name}_STGCN_WAVE_{datetime.now().strftime('%y%m%d_%H%M%S')}"
    )
    training_device = get_auto_device()

    logger.info(f"Training Name: {run_name}")
    logger.info(f"Training Device --> {training_device}")

    wandb.init(project="METR-IMC", name=run_name, config=asdict(config))
    wandb_config: Config = wandb.config
    wandb_config.update({"device": str(training_device)})

    fix_seed(config.seed)

    adj_mx_raw = import_adj_mx(config.adj_mx_filepath)
    sparse_mx = sp.coo_matrix(adj_mx_raw.adj_mx)
    G = dgl.from_scipy(sparse_mx)
    G = G.to(training_device)

    dataset = MetrDataset.from_file(
        config.tsfilepath,
        config.window,
        config.pred_len,
        config.missing_labels_filepath,
    )
    train_subset, val_subset, test_subset, scaler = dataset.split(
        train_ratio=config.train_ratio, valid_ratio=config.valid_ratio
    )

    train_iter = DataLoader(
        train_subset, batch_size=config.batch_size, collate_fn=MetrDataset.collate_fn
    )
    valid_iter = DataLoader(
        val_subset, batch_size=config.batch_size, collate_fn=MetrDataset.collate_fn
    )
    test_iter = DataLoader(
        test_subset,
        batch_size=config.batch_size,
        collate_fn=MetrDataset.collate_fn_with_missing,
    )

    model = STGCN_WAVE(
        config.channels,  # Blocks
        config.window,  # History Length
        dataset.num_nodes,  # Number of Roads(Routes)
        G,
        config.drop_rate,  # Dropout Rate (Normally 0)
        config.num_layers,
        training_device,  # Device
        config.control_str,
    )
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **config.scheduler)

    wandb.watch(model)

    min_valid_loss = np.inf
    for epoch in range(1, config.epochs + 1):
        print(f"Epoch {epoch}/{config.epochs}")
        epoch_loss = __train_model(
            train_iter, model, loss_fn, optimizer, training_device
        )
        scheduler.step()  # Update at new epoch
        valid_loss = __validate_model(valid_iter, model, loss_fn, training_device)

        log_content = {
            "epoch": epoch,
            "epoch_loss": epoch_loss,
            "val_loss": valid_loss,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        logger.info(f"Training Result:\r\n{log_content}")

        if valid_loss < min_valid_loss:  # When finding the best model
            logger.info("Best Model Found!")
            min_valid_loss = valid_loss
            torch.save(model.state_dict(), config.savemodelpath)
            MAE, MAPE, RMSE = evaluate_model_(test_iter, scaler, model, training_device)
            eval_result = {
                "epoch_MAE": MAE,
                "epoch_RMSE": RMSE,
                "epoch_MAPE": MAPE,
            }
            logger.info(f"Best Model Result:\r\n{eval_result}")
            log_content.update(eval_result)

        wandb.log(log_content)

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

    MAE, MAPE, RMSE = evaluate_model_(test_iter, scaler, model, training_device)
    test_result = {"test_MAE": MAE, "test_RMSE": RMSE, "test_MAPE": MAPE}
    logger.info(f"Test Result:\r\n{test_result}")
    wandb.log(test_result)
    wandb.finish()


def __train_model(
    dataloader: DataLoader,
    model: nn.Module,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
):
    model = model.to(device)
    model.train()

    loss_sum = 0.0
    for x, y in tqdm(dataloader):
        x: Tensor = x.to(device)
        y: Tensor = y.to(device)
        y_pred: Tensor = model(x)
        y_pred = y_pred.view(len(x), -1)
        loss: Tensor = loss_fn(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * y.shape[0]
    loss_ave = loss_sum / len(dataloader.dataset)

    return loss_ave


def __validate_model(
    dataloader: DataLoader,
    model: nn.Module,
    loss_fn: nn.Module,
    device: torch.device,
):
    model = model.to(device)
    model.eval()

    loss_sum = 0.0
    with torch.no_grad():
        for x, y in tqdm(dataloader):
            # Todo: Check if missing_value is correctly handled
            x: Tensor = x.to(device)
            y: Tensor = y.to(device)
            y_pred: Tensor = model(x)
            y_pred = y_pred.view(len(x), -1)
            loss: Tensor = loss_fn(y_pred, y)
            loss_sum += loss.item() * y.shape[0]
    loss_ave = loss_sum / len(dataloader.dataset)

    return loss_ave


# Todo: 데이터셋에서 결측치 여부에 대한 레이블링을 추가
# Todo: collate_fn을 다르게하여 결측치 레이블을 추가할지 말지를 결정


## -------------------------------------- ##


def train(config: HyperParams):
    run_name = (
        f"{config.dataset_name}_STGCN_WAVE_{datetime.now().strftime('%y%m%d_%H%M%S')}"
    )
    training_device = get_auto_device()
    wandb.init(project="METR-IMC", name=run_name, config=asdict(config))
    wandb.config.update({"device": str(training_device)})

    with open(config.sensorsfilepath) as f:
        sensor_ids = f.read().strip().split(",")
    distance_df = pd.read_csv(config.disfilepath, dtype={"from": "str", "to": "str"})

    adj_mx = get_adjacency_matrix(distance_df, sensor_ids)
    sp_mx = sp.coo_matrix(adj_mx)
    G = dgl.from_scipy(sp_mx)

    df = pd.read_hdf(config.tsfilepath)
    num_samples, num_nodes = df.shape

    # Hyperparameters
    n_his = config.window
    save_path = config.savemodelpath
    n_pred = config.pred_len
    n_route = num_nodes
    blocks = config.channels
    drop_prob = 0
    num_layers = config.num_layers
    batch_size = config.batch_size
    epochs = config.epochs
    lr = config.lr

    os.makedirs(os.path.split(save_path)[0], exist_ok=True)

    len_val = round(num_samples * 0.1)
    len_train = round(num_samples * 0.7)
    train = df[:len_train]
    val = df[len_train : len_train + len_val]
    test = df[len_train + len_val :]

    scaler = StandardScaler()
    train = scaler.fit_transform(train)
    val = scaler.transform(val)
    test = scaler.transform(test)

    x_train, y_train = data_transform(train, n_his, n_pred, training_device)
    x_val, y_val = data_transform(val, n_his, n_pred, training_device)
    x_test, y_test = data_transform(test, n_his, n_pred, training_device)

    train_data = TensorDataset(x_train, y_train)
    train_iter = DataLoader(train_data, batch_size, shuffle=True)
    val_data = TensorDataset(x_val, y_val)
    val_iter = DataLoader(val_data, batch_size)
    test_data = TensorDataset(x_test, y_test)
    test_iter = DataLoader(test_data, batch_size)

    loss = nn.MSELoss()
    G = G.to(training_device)
    model = STGCN_WAVE(
        blocks,
        n_his,
        n_route,
        G,
        drop_prob,
        num_layers,
        training_device,
        config.control_str,
    ).to(training_device)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)

    wandb.watch(model)

    min_val_loss = np.inf
    for epoch in range(1, epochs + 1):
        l_sum, n = 0.0, 0
        model.train()
        for x, y in tqdm(train_iter, desc=f"Epoch {epoch}/{epochs}"):
            y_pred = model(x).view(len(x), -1)
            l: torch.Tensor = loss(y_pred, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]
            wandb.log({"train_loss": l})
        scheduler.step()
        epoch_loss = l_sum / n
        val_loss = evaluate_model(model, loss, val_iter)

        gpu_mem_alloc = (
            torch.cuda.max_memory_allocated() / 1000000
            if torch.cuda.is_available()
            else 0
        )
        gpu_mem_alloc = (
            torch.mps.current_allocated_memory() / 1000000
            if torch.backends.mps.is_available()
            else gpu_mem_alloc
        )

        wandb.log(
            {
                "epoch": epoch + 1,
                "epoch_loss": epoch_loss,
                "val_loss": val_loss,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "gpu_memory": gpu_mem_alloc,
            }
        )

        if val_loss < min_val_loss:
            min_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
        print(
            "epoch",
            epoch,
            ", train loss:",
            l_sum / n,
            ", validation loss:",
            val_loss,
        )

    best_model = STGCN_WAVE(
        blocks,
        n_his,
        n_route,
        G,
        drop_prob,
        num_layers,
        training_device,
        config.control_str,
    ).to(training_device)
    best_model.load_state_dict(torch.load(save_path))

    l = evaluate_model(best_model, loss, test_iter)
    MAE, MAPE, RMSE = evaluate_metric(best_model, test_iter, scaler)
    print("test loss:", l, "\nMAE:", MAE, ", MAPE:", MAPE, ", RMSE:", RMSE)
    wandb.log({"test_MAE": MAE, "test_RMSE": RMSE, "test_MAPE": MAPE})
    wandb.finish()
