import os
from typing import List, Optional

import lightning as L
from lightning import Trainer
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import WandbLogger
from metr.components import Metadata
import numpy as np
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader

from songdo_llm.dataset import BaseTrafficDataModule
from songdo_llm.model.lightning.traffic_prediction import TrafficVolumePredictionModule

# Parameters
epochs = 1

metadata_path = "../datasets/metr-imc/metadata.h5"
output_dir = "./output/simple_predictions"
os.makedirs(output_dir, exist_ok=True)

print(f"Starting simple prediction...-->")


data_module = BaseTrafficDataModule()
traffic_model = TrafficVolumePredictionModule()
wandb_logger = WandbLogger(project="Songdo_LLM", log_model="all")
trainer = Trainer(
    max_epochs=epochs,
    accelerator="auto",
    devices="auto",
    log_every_n_steps=1,
    default_root_dir=output_dir,
    logger=wandb_logger,
    callbacks=[
        EarlyStopping(monitor="val_loss", mode="min", patience=5),
        ModelCheckpoint(
            dirpath=output_dir,
            filename="best-{epoch:02d}-{val_loss:.2f}",
            save_top_k=1,
            monitor="val_loss",
            mode="min",
        ),
        LearningRateMonitor(logging_interval="step"),
    ],
)


# ------------ Training ------------
def test_training():
    final_train_loss: Optional[float] = None
    final_valid_loss: Optional[float] = None

    metadata = Metadata.import_from_hdf(metadata_path)
    speed_limit_map = metadata.data.set_index("LINK_ID")["MAX_SPD"].to_dict()
    lane_map = metadata.data.set_index("LINK_ID")["LANES"].to_dict()

    trainer.fit(traffic_model, datamodule=data_module)

    final_train_loss = trainer.callback_metrics.get("train_loss")
    final_valid_loss = trainer.callback_metrics.get("val_loss")

    print(f"Final Train Loss: {final_train_loss}")
    print(f"Final Validation Loss: {final_valid_loss}")


# ------------ Evaluation ------------
def evaluate_model(
    model: L.LightningModule,
    trainer: L.Trainer,
    scaler: MinMaxScaler,
    dataloader: DataLoader,
):
    train_predictions = trainer.predict(model, dataloaders=dataloader)

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


def test_evaluation():
    test_pred, test_true = evaluate_model(
        traffic_model,
        trainer,
        data_module.scaler,
        data_module.predict_dataloader(),
    )

    test_mae = mean_absolute_error(test_true, test_pred)
    test_rmse = root_mean_squared_error(test_true, test_pred)

    print(f"Test MAE: {test_mae:.6f}")
    print(f"Test RMSE: {test_rmse:.6f}")
