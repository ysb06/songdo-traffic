import logging
import os

import geopandas as gpd

from metr.components import (
    AdjacencyMatrix,
    DistancesImc,
    IdList,
    Metadata,
    SensorLocations,
    TrafficData,
)
from metr.components.metr_imc.outlier import (
    HourlyInSensorZscoreOutlierProcessor,
    HourlyZscoreOutlierProcessor,
    RemovingWeirdZeroOutlierProcessor,
    SimpleAbsoluteOutlierProcessor,
    SimpleZscoreOutlierProcessor,
    TrafficCapacityAbsoluteOutlierProcessor,
)
from metr.processors import *

logger = logging.getLogger(__name__)

SUBSET_TRAFFIC_TRAINING_PATH = os.path.join(SUBSET_DATASET_ROOT_DIR, TRAFFIC_TRAINING_FILENAME)
SUBSET_METADATA_PATH = os.path.join(SUBSET_DATASET_ROOT_DIR, METADATA_FILENAME)


def run_process():
    os.makedirs(OUTLIER_PROCESSED_DIR, exist_ok=True)
    
    do_all_outlier_removing()


def do_all_outlier_removing():
    traffic_data = TrafficData.import_from_hdf(SUBSET_TRAFFIC_TRAINING_PATH, dtype=float)
    metadata = Metadata.import_from_hdf(SUBSET_METADATA_PATH)
    speed_limit_map = metadata.data.set_index("LINK_ID")["MAX_SPD"].to_dict()
    lane_map = metadata.data.set_index("LINK_ID")["LANES"].to_dict()

    processor_set = [
        (
            [
                RemovingWeirdZeroOutlierProcessor(),
                SimpleAbsoluteOutlierProcessor(8000),
            ],
            "absolute_simple",
        ),
        (
            [
                RemovingWeirdZeroOutlierProcessor(),
                TrafficCapacityAbsoluteOutlierProcessor(speed_limit_map, lane_map),
            ],
            "absolute_traffic_capacity",
        ),
        (
            [
                RemovingWeirdZeroOutlierProcessor(),
                SimpleZscoreOutlierProcessor(5.0),
            ],
            "zscore_simple",
        ),
        (
            [
                RemovingWeirdZeroOutlierProcessor(),
                HourlyZscoreOutlierProcessor(5.0),
            ],
            "zscore_hourly_all",
        ),
        (
            [
                RemovingWeirdZeroOutlierProcessor(),
                HourlyInSensorZscoreOutlierProcessor(5.0),
            ],
            "zscore_hourly_in_sensor",
        ),
    ]

    for processors, name in processor_set:
        traffic_data.reset_data()
        traffic_data.remove_outliers(processors)
        traffic_data.to_hdf(os.path.join(OUTLIER_PROCESSED_DIR, f"{name}.h5"))
