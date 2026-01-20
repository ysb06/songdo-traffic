import logging
from typing import List, Tuple

import torch

from metr.components.adj_mx import AdjacencyMatrix
from metr.components.metr_imc.interpolation import Interpolator
from metr.components.metr_imc.interpolation.bgcp import BGCPInterpolator
from metr.components.metr_imc.interpolation.brits import BRITSInterpolator
from metr.components.metr_imc.interpolation.grin import GRINInterpolator
from metr.components.metr_imc.interpolation.knn import SpatialKNNInterpolator
from metr.components.metr_imc.interpolation.mice import SpatialMICEInterpolator
from metr.components.metr_imc.interpolation.trmf import TRMFInterpolator
from metr.components.metr_imc.outlier import OutlierProcessor
from metr.components.metr_imc.outlier.base import SimpleAbsoluteOutlierProcessor
from metr.pipeline import generate_raw_dataset, generate_subset
from metr.utils import PathConfig

logging.basicConfig(
    format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
    datefmt="%Y/%m/%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Generate Raw Datasets
# generate_raw_dataset()

# Generate Base Subset Datasets (테스트 용)
base_subset_path_conf = PathConfig.from_yaml("../config_base.yaml")
# generate_subset(
#     subset_path_conf=base_subset_path_conf,
#     target_data_start="2023-01-26 00:00:00",
#     cluster_count=1,
#     missing_rate_threshold=0.9,
# )


# Generate Data Interpolation Subset
def generate_interpolated_subset(key: str, interpolator: Interpolator):
    subset_path_conf = PathConfig.from_yaml(f"../config_{key}.yaml")
    interpolation_processors: List[Interpolator] = [
        interpolator,
    ]

    generate_subset(
        subset_path_conf=subset_path_conf,
        target_data_start="2023-01-26 00:00:00",
        cluster_count=1,
        missing_rate_threshold=0.9,
        interpolation_processors=interpolation_processors,
    )


raw_path_conf = PathConfig.from_yaml("../config.yaml")
base_adj_mx = AdjacencyMatrix.import_from_pickle(base_subset_path_conf.adj_mx_path)
base_adj_mx_tensor = torch.tensor(base_adj_mx.adj_mx, dtype=torch.float32)

interpolation_processors: List[Tuple[str, Interpolator]] = [
    # ("mice", SpatialMICEInterpolator(base_adj_mx)),
    # ("knn", SpatialKNNInterpolator(base_adj_mx)),
    # ("bgcp", BGCPInterpolator()),
    # ("trmf", TRMFInterpolator()),
    # ("brits", BRITSInterpolator()),
    ("grin", GRINInterpolator(base_adj_mx_tensor)),
]
for key, interpolator in interpolation_processors:
    logger.info(f'Generating interpolated subset with "{key}" interpolator.')
    generate_interpolated_subset(key, interpolator)
