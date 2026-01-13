from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import WandbLogger
from metr.datasets.rnn.datamodule import MultiSensorTrafficDataModule

from .models.rnn.module import MultiSensorLSTMLightningModule


def main(dataset_path: str = "./data/selected_small_v1/metr-imc.h5"):
    data = MultiSensorTrafficDataModule(dataset_path)
    data.setup()  # Setup을 먼저 호출하여 scaler 생성
    
    model = MultiSensorLSTMLightningModule(scaler=data.scaler)

    output_dir = "./output/lstm"
    wandb_logger = WandbLogger(project="Traffic-IMC-LSTM", log_model="all")

    trainer = Trainer(
        max_epochs=100,
        accelerator="auto",
        devices="auto",
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
    trainer.fit(model, data)
    trainer.test(model, data)