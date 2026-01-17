from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import WandbLogger
from metr.datasets.rnn.datamodule import MultiSensorTrafficDataModule

from .models.rnn.module import MultiSensorLSTMLightningModule
from metr.utils import PathConfig


def main(path_config: PathConfig):
    data = MultiSensorTrafficDataModule(
        path_config.metr_imc_training_path,
        path_config.metr_imc_test_path,
        path_config.metr_imc_test_missing_path,
    )
    data.setup()  # Setup을 먼저 호출하여 scaler 생성

    model = MultiSensorLSTMLightningModule(scaler=data.scaler)

    output_dir = "./output/lstm"
    wandb_logger = WandbLogger(
        name="LSTM-MICE-00", project="IMC-Traffic", log_model="all"
    )

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
