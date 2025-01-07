from typing import List, Optional, Tuple

import lightning as L
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader
from .dataset import TrafficDataModule


def evaluate_model(
    model: L.LightningModule,
    trainer: L.Trainer,
    data_module: TrafficDataModule,
) -> Tuple[np.ndarray, np.ndarray]:
    dataloaders = data_module.predict_dataloader()
    scaler = data_module.scaler

    train_predictions = trainer.predict(model, dataloaders=dataloaders)

    train_preds_list: List[np.ndarray] = []
    train_true_list: List[np.ndarray] = []

    for preds, y_batch in train_predictions:
        train_preds_list.append(preds)
        train_true_list.append(y_batch)
    train_preds_arr = np.concatenate(train_preds_list, axis=0)
    train_true_arr = np.concatenate(train_true_list, axis=0)

    train_preds_inv = scaler.inverse_transform(train_preds_arr)
    train_y_inv = scaler.inverse_transform(train_true_arr)

    return train_preds_inv.squeeze(1), train_y_inv.squeeze(1)


