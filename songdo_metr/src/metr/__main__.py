import logging

from metr.pipeline import generate_raw_dataset
from metr.utils import PathConfig

logging.basicConfig(
    format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
    datefmt="%Y/%m/%d %H:%M:%S",
    level=logging.INFO,
)

# generate_raw_dataset()
config = PathConfig.from_yaml("config.yaml")
print(config)
