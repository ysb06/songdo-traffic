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


def main(name_key: str, path_config: PathConfig, code: int = 0):
    """Main training function for LSTM.
    
    Args:
        name_key: Name key for WandB logging (e.g., 'KNN', 'MICE')
        path_config: PathConfig instance with dataset paths
        code: Run code number for identification
    """
    data = MultiSensorTrafficDataModule(
        path_config.metr_imc_training_path,
        path_config.metr_imc_test_path,
        path_config.metr_imc_test_missing_path,
    )
    data.setup()  # Setup을 먼저 호출하여 scaler 생성

    model = MultiSensorLSTMLightningModule(scaler=data.scaler)

    output_dir = f"./output/lstm/{name_key}_{code:02d}"
    wandb_logger = WandbLogger(
        name=f"LSTM-{name_key}-{code:02d}", project="IMC-Traffic", log_model="all"
    )

    trainer = Trainer(
        max_epochs=20,
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
        precision="16-mixed"
    )
    trainer.fit(model, data)
    trainer.test(model, data)


if __name__ == "__main__":
    import os
    from .utils import parse_training_args, get_config_path
    
    args = parse_training_args()
    
    # GPU 설정 (상위 프로세스에서 설정되지 않은 경우에만)
    # __main__.py에서 subprocess로 실행 시 이미 CUDA_VISIBLE_DEVICES가 설정됨
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    
    # Config 로드 및 실행
    config_path = get_config_path(args.data)
    path_config = PathConfig.from_yaml(config_path)
    main(args.data, path_config, code=args.code)
