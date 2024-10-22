import logging
from dataclasses import asdict
from datetime import datetime

import dgl
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import wandb
from metr.components.adj_mx import AdjacencyMatrix
from metr.dataloader import MetrDataset
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm
from wandb import Config

from .model import STGCN_WAVE
from .test import evaluate_model_
from .utils import HyperParams, fix_seed, get_auto_device

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

    adj_mx_raw = AdjacencyMatrix.import_from_pickle(config.adj_mx_filepath)
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
        epoch_loss = _train_model(
            train_iter, model, loss_fn, optimizer, training_device
        )
        scheduler.step()  # Update at new epoch
        valid_loss = _validate_model(valid_iter, model, loss_fn, training_device)

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


def _train_model(
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


def _validate_model(
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