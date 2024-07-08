# Load configuration
import pytorch_lightning as pl
import logging
import yaml

from songdo_traffic_core.dataloader.stgcn import STGCNDataModule
from songdo_traffic_core.models.stgcn.earlystopping import EarlyStopping
from songdo_traffic_core.trainers.stgcn.trainer import STGCN, set_env

import argparse
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="STGCN")
parser.add_argument(
    "--config",
    default="./configs/stgcn/config_base.yaml",
    help="Config path for training and testing STGCN model",
)
args = parser.parse_args()

logger.info(f"Loading configuration from {args.config}")
config = OmegaConf.load(args.config)
print(config)


# # Set environment
# set_env(config.seed)

# # Set device
# if config.enable_cuda and torch.cuda.is_available():
#     config.device = torch.device("cuda")
#     torch.cuda.empty_cache()
# else:
#     config.device = torch.device("cpu")

# # Prepare data
# data_module = STGCNDataModule(config)

# # Prepare model
# Ko = config.n_his - (config.Kt - 1) * 2 * config.stblock_num
# blocks = []
# blocks.append([1])
# for l in range(config.stblock_num):
#     blocks.append([64, 16, 64])
# if Ko == 0:
#     blocks.append([128])
# elif Ko > 0:
#     blocks.append([128, 128])
# blocks.append([1])

# model = STGCN(config, blocks, data_module.n_vertex)

# # Prepare trainer
# early_stop_callback = EarlyStopping(
#     monitor="val_loss",
#     min_delta=0.0,
#     patience=config.patience,
#     verbose=True,
#     mode="min",
# )

# trainer = pl.Trainer(
#     max_epochs=config.epochs,
#     callbacks=[early_stop_callback],
#     gpus=1 if config.enable_cuda and torch.cuda.is_available() else 0,
# )

# # Train and test
# trainer.fit(model, data_module)
# trainer.test(model, data_module)
