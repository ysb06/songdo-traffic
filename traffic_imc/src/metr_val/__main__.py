from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import WandbLogger
from metr.datasets.stgcn.datamodule import STGCNDataModule

from .models.rnn.module import LSTMLightningModule


def main():
    data = STGCNDataModule("./data/selected_small_v1")
    data.setup()
    print(data)


if __name__ == "__main__":
    main()
