import glob
import os
from pathlib import Path

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import WandbLogger

# 데이터 모듈과 모델 임포트
from metr.datasets.rnn.datamodule import MultiSensorTrafficDataModule

from ..models.rnn.module import MultiSensorLSTMLightningModule
from . import MODEL_OUTPUT_DIR
from metr.datasets.rnn.dataloader import collate_multi_sensor_simple


def find_latest_model_checkpoint(model_output_dir: Path) -> Path:
    """MODEL_OUTPUT_DIR에서 최신 체크포인트 파일을 찾습니다."""
    if not model_output_dir.exists():
        return None

    # 모든 체크포인트 파일 패턴을 찾습니다
    checkpoint_patterns = [
        # model_output_dir / "**" / "*.ckpt",
        model_output_dir
        / "**"
        / "best-*.ckpt",
    ]

    all_checkpoints = []
    for pattern in checkpoint_patterns:
        all_checkpoints.extend(glob.glob(str(pattern), recursive=True))

    if not all_checkpoints:
        return None

    # 파일 수정 시간을 기준으로 최신 파일을 찾습니다
    latest_checkpoint = max(all_checkpoints, key=os.path.getmtime)
    return Path(latest_checkpoint)


def load_or_train_model(
    model_output_dir: Path = MODEL_OUTPUT_DIR,
    dataset_path: str = "./data/selected_small_v1/metr-imc.h5",
) -> MultiSensorLSTMLightningModule:
    """최신 모델을 로드하거나 새로 훈련합니다."""

    # 최신 체크포인트 찾기
    latest_checkpoint = find_latest_model_checkpoint(model_output_dir)

    if latest_checkpoint:
        print(f"최신 모델 발견: {latest_checkpoint}")
        print("기존 모델을 로드합니다...")
    else:
        print("기존 모델이 없습니다. 새로운 모델을 훈련합니다...")
        model_training(model_output_dir, dataset_path)
        latest_checkpoint = find_latest_model_checkpoint(model_output_dir)
        if latest_checkpoint is None:
            raise FileNotFoundError("모델 훈련 후에도 체크포인트를 찾을 수 없습니다.")

    # 체크포인트에서 모델 로드
    model = MultiSensorLSTMLightningModule.load_from_checkpoint(str(latest_checkpoint))
    print("모델 로드 완료!")
    return model


def model_training(
    model_output_dir: Path,
    dataset_path: str,
    batch_size: int = 128,
    num_workers: int = 9,
    scale_method: str = "none",
):
    rnn_data = MultiSensorTrafficDataModule(
        dataset_path,
        batch_size=batch_size,
        num_workers=num_workers,
        scale_method=scale_method,
    )
    rnn_model = MultiSensorLSTMLightningModule()

    wandb_logger = WandbLogger(project="Traffic-IMC", log_model="all")

    trainer = Trainer(
        max_epochs=100,
        accelerator="auto",
        devices="auto",
        default_root_dir=model_output_dir,
        logger=wandb_logger,
        callbacks=[
            EarlyStopping(monitor="val_loss", mode="min", patience=5),
            ModelCheckpoint(
                dirpath=model_output_dir,
                filename="best-{epoch:02d}-{val_loss:.2f}",
                save_top_k=1,
                monitor="val_loss",
                mode="min",
            ),
            LearningRateMonitor(logging_interval="step"),
        ],
    )
    trainer.fit(rnn_model, rnn_data)
    trainer.test(rnn_model, rnn_data)

    return rnn_model


def sanitize_sensor_name(sensor_name: str) -> str:
    """센서명을 안전한 키로 변환합니다.

    Args:
        sensor_name: 원본 센서명

    Returns:
        안전한 키 문자열
    """
    return f"sensor_{sensor_name.replace('-', '_').replace('.', '_')}"


def load_config(config_path: str = None) -> dict:
    """YAML 설정 파일을 로드합니다.

    Args:
        config_path: 설정 파일 경로 (기본값: 현재 디렉토리의 config.yaml)

    Returns:
        설정 딕셔너리
    """
    import yaml
    from pathlib import Path

    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config
