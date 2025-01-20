import logging
import os
from typing import Optional
from metr.components.metr_imc.outlier import (
    RemovingWeirdZeroOutlierProcessor,
    TrafficCapacityAbsoluteOutlierProcessor,
    HourlyInSensorZscoreOutlierProcessor,
    HourlyZscoreOutlierProcessor,
    MADOutlierProcessor,
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
    outlier_set_1_filename = "non_simple.h5"
    outlier_sets.append((outlier_set_1, outlier_set_1_filename))

    outlier_set_2 = [
        RemovingWeirdZeroOutlierProcessor(),
        TrafficCapacityAbsoluteOutlierProcessor(speed_limit_map, lane_map),
        HourlyInSensorZscoreOutlierProcessor(),
    ]
    outlier_set_2_filename = "zsc_hrsnr.h5"
    outlier_sets.append((outlier_set_2, outlier_set_2_filename))

    outlier_set_3 = [
        RemovingWeirdZeroOutlierProcessor(),
        TrafficCapacityAbsoluteOutlierProcessor(speed_limit_map, lane_map),
        HourlyZscoreOutlierProcessor(),
    ]
    outlier_set_3_filename = "zsc_hrsimp.h5"
    outlier_sets.append((outlier_set_3, outlier_set_3_filename))

    outlier_set_4 = [
        RemovingWeirdZeroOutlierProcessor(),
        TrafficCapacityAbsoluteOutlierProcessor(speed_limit_map, lane_map),
        MADOutlierProcessor(),
    ]
    outlier_set_4_filename = "zsc_madev.h5"
    outlier_sets.append((outlier_set_4, outlier_set_4_filename))

    # 이상치는 일반적으로 평활화로 처리하는 것을 보임
    # 평활화는 데이터의 불안정하고 불규칙한 특성을 제거해 주기도 하지만 돌발상황과 같은 특별한 정보를 제거해 버릴 수 도 있음.
    # 즉, 평활화는 특별한 목적이 있을 때, 해당 목적에 맞게 알고리즘을 선택하고 적용해야 함.
    # 본 논문에서는 이러한 특별한 정보를 최대한 보존하면서 이상치를 제거하는 데 초점을 맞춤 (추후 데이터셋 제작 시 중요)


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
