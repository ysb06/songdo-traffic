import logging
import os
from typing import Optional
from metr.components.metr_imc.outlier import (
    TrafficCapacityAbsoluteOutlierProcessor,
    RemovingWeirdZeroOutlierProcessor,
    SimpleZscoreOutlierProcessor,
    HourlyInSensorZscoreOutlierProcessor,
)
from metr.components.metr_imc import TrafficData
from metr.components.metadata import Metadata

logger = logging.getLogger(__name__)

TRAFFIC_RAW_PATH = "../datasets/metr-imc/metr-imc.h5"
METADATA_RAW_PATH = "../datasets/metr-imc/metadata.h5"
OUTPUT_DIR = "./output/outlier_processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def process_outlier(
    start_datetime: Optional[str] = "2024-01-01 00:00:00",
    end_datetime: Optional[str] = "2024-08-31 23:00:00",
):
    metadata = Metadata.import_from_hdf(METADATA_RAW_PATH)
    speed_limit_map = metadata.data.set_index("LINK_ID")["MAX_SPD"].to_dict()
    lane_map = metadata.data.set_index("LINK_ID")["LANES"].to_dict()

    outlier_sets = []

    outlier_set_1 = [
        RemovingWeirdZeroOutlierProcessor(),
        TrafficCapacityAbsoluteOutlierProcessor(speed_limit_map, lane_map),
    ]
    outlier_set_1_filename = "abs_capacity.h5"
    outlier_sets.append((outlier_set_1, outlier_set_1_filename))

    outlier_set_2 = [
        RemovingWeirdZeroOutlierProcessor(),
        SimpleZscoreOutlierProcessor(5.0),
    ]
    outlier_set_2_filename = "zsc_simple.h5"
    outlier_sets.append((outlier_set_2, outlier_set_2_filename))

    outlier_set_3 = [
        RemovingWeirdZeroOutlierProcessor(),
        HourlyInSensorZscoreOutlierProcessor(5.0),
    ]
    outlier_set_3_filename = "zsc_hourly.h5"
    outlier_sets.append((outlier_set_3, outlier_set_3_filename))

    for outlier_set, filename in outlier_sets:
        traffic_data = TrafficData.import_from_hdf(TRAFFIC_RAW_PATH, dtype=float)
        if start_datetime is not None:
            traffic_data.start_time = start_datetime
        if end_datetime is not None:
            traffic_data.end_time = end_datetime
        traffic_data.remove_outliers(outlier_set)
        traffic_data.to_hdf(os.path.join(OUTPUT_DIR, filename))

        logger.info(f"Processed {filename} from {TRAFFIC_RAW_PATH}")
        if start_datetime is not None or end_datetime is not None:
            logger.info(
                f"Time range: From {traffic_data.data.index[0]} to {traffic_data.data.index[-1]}"
            )
