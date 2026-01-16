import logging
import pickle
from typing import List

from metr.components.metr_imc.interpolation import Interpolator
from metr.components.metr_imc.interpolation.mice import SpatialMICEInterpolator
from metr.components.metr_imc.interpolation.knn import SpatialKNNInterpolator
from metr.components.metr_imc.interpolation.bgcp import BGCPInterpolator
from metr.components.metr_imc.outlier import OutlierProcessor
from metr.components.metr_imc.outlier.base import SimpleAbsoluteOutlierProcessor
from metr.pipeline import generate_raw_dataset, generate_subset
from metr.utils import PathConfig
from metr.components.adj_mx import AdjacencyMatrix

logging.basicConfig(
    format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
    datefmt="%Y/%m/%d %H:%M:%S",
    level=logging.INFO,
)

# # Generate Raw Datasets
# generate_raw_dataset()

# # Generate Base Subset Datasets
# generate_subset(
#     target_data_start="2023-01-26 00:00:00",
#     cluster_count=1,
#     missing_rate_threshold=0.9,
# )

# Generate Final Subset Datasets with Data Correction
raw_path_conf = PathConfig.from_yaml("../config.yaml")
subset_path_conf = PathConfig.from_yaml("../config_bgcp.yaml")
adj_mx = AdjacencyMatrix.import_from_pickle(raw_path_conf.adj_mx_path)

outlier_processors: List[OutlierProcessor] = [
    SimpleAbsoluteOutlierProcessor(threshold=3450),  # Example threshold
]
interpolation_processors: List[Interpolator] = [
    # SpatialKNNInterpolator(adj_mx),
    # SpatialMICEInterpolator(adj_mx),
    BGCPInterpolator(),
]

generate_subset(
    subset_path_conf=subset_path_conf,
    target_data_start="2023-01-26 00:00:00",
    cluster_count=1,
    missing_rate_threshold=0.9,
    outlier_processors=outlier_processors,
    interpolation_processors=interpolation_processors,
)
