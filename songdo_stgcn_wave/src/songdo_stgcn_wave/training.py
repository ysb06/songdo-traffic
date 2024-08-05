import argparse
import gc
import os
import random
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import dgl
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
import wandb
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .utils import Config, get_auto_device
from stgcn_wave.sensors2graph import get_adjacency_matrix
from stgcn_wave.model import STGCN_WAVE
from stgcn_wave.load_data import data_transform
from stgcn_wave.utils import evaluate_model, evaluate_metric


def train(config: Config):
    run_name = (
        f"{config.dataset_name}_STGCN_WAVE_{datetime.now().strftime('%y%m%d_%H%M%S')}"
    )
    training_divice = get_auto_device()
    wandb.init(project='METR-IMC', name=run_name, config=asdict(config))
    wandb.config.update({"device": str(training_divice)})

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

    x_train, y_train = data_transform(train, n_his, n_pred, training_divice)
    x_val, y_val = data_transform(val, n_his, n_pred, training_divice)
    x_test, y_test = data_transform(test, n_his, n_pred, training_divice)

    train_data = TensorDataset(x_train, y_train)
    train_iter = DataLoader(train_data, batch_size, shuffle=True)
    val_data = TensorDataset(x_val, y_val)
    val_iter = DataLoader(val_data, batch_size)
    test_data = TensorDataset(x_test, y_test)
    test_iter = DataLoader(test_data, batch_size)

    loss = nn.MSELoss()
    G = G.to(training_divice)
    model = STGCN_WAVE(
        blocks,
        n_his,
        n_route,
        G,
        drop_prob,
        num_layers,
        training_divice,
        config.control_str,
    ).to(training_divice)
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
        blocks, n_his, n_route, G, drop_prob, num_layers, training_divice, config.control_str
    ).to(training_divice)
    best_model.load_state_dict(torch.load(save_path))

    l = evaluate_model(best_model, loss, test_iter)
    MAE, MAPE, RMSE = evaluate_metric(best_model, test_iter, scaler)
    print("test loss:", l, "\nMAE:", MAE, ", MAPE:", MAPE, ", RMSE:", RMSE)
    wandb.log({"test_MAE": MAE, "test_RMSE": RMSE, "test_MAPE": MAPE})
    wandb.finish()