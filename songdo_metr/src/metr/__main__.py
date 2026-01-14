import logging
from typing import List

from metr.components.metr_imc.interpolation import Interpolator
from metr.components.metr_imc.interpolation.mice import MICEInterpolator
from metr.components.metr_imc.outlier import OutlierProcessor
from metr.components.metr_imc.outlier.base import SimpleAbsoluteOutlierProcessor
from metr.pipeline import generate_raw_dataset, generate_subset
from metr.utils import PathConfig

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
outlier_processors: List[OutlierProcessor] = [
    SimpleAbsoluteOutlierProcessor(threshold=3450),  # Example threshold
]
interpolation_processors: List[Interpolator] = [
    MICEInterpolator(),
]
mice_subset_path_conf = PathConfig.from_yaml("../config_mice.yaml")
generate_subset(
    subset_path_conf=mice_subset_path_conf,
    target_data_start="2023-01-26 00:00:00",
    cluster_count=1,
    missing_rate_threshold=0.9,
    outlier_processors=outlier_processors,
    interpolation_processors=interpolation_processors,
)
