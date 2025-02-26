import logging
import os
from typing import List, Optional
from copy import deepcopy
from glob import glob

from matplotlib import pyplot as plt
from metr.components.metadata import Metadata
import pandas as pd
import yaml

# 센서 간 관계로부터 이상치를 판단하는 방법은 제외
# 해당 방법은 다른 연구를 통해서 좀 더 깊게 연구한다고 명시할 것
from metr.components.metr_imc import TrafficData
from metr.components.metr_imc.outlier import (
    OutlierProcessor,
    HourlyInSensorZscoreOutlierProcessor,
    InSensorZscoreOutlierProcessor,
    MADOutlierProcessor,
    RemovingWeirdZeroOutlierProcessor,
    TrafficCapacityAbsoluteOutlierProcessor,
    TrimmedMeanOutlierProcessor,
    WinsorizedOutlierProcessor,
)

logger = logging.getLogger(__name__)

TRAFFIC_RAW_PATH = "../datasets/metr-imc/metr-imc.h5"
METADATA_RAW_PATH = "../datasets/metr-imc/metadata.h5"
OUTPUT_DIR = "./output/outlier_processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def remove_base_outliers(
    start_datetime: Optional[str] = "2023-12-01 00:00:00",
    end_datetime: Optional[str] = "2024-08-31 23:00:00",
    traffic_capacity_adjustment_rate: float = 1.5,
    raw_path: str = TRAFFIC_RAW_PATH,
    metadata_path: str = METADATA_RAW_PATH,
    output_dir: str = OUTPUT_DIR,
    remove_empty: bool = True,
):
    raw = TrafficData.import_from_hdf(raw_path).data
    metadata = Metadata.import_from_hdf(metadata_path)
    speed_limit_map = metadata.data.set_index("LINK_ID")["MAX_SPD"].to_dict()
    lane_map = metadata.data.set_index("LINK_ID")["LANES"].to_dict()

    base_proc_1 = RemovingWeirdZeroOutlierProcessor()
    base_proc_2 = TrafficCapacityAbsoluteOutlierProcessor(
        speed_limit_map,
        lane_map,
        adjustment_rate=traffic_capacity_adjustment_rate,
    )

    base_data = raw.copy()
    proc_1_data = base_proc_1.process(base_data)
    proc_2_data = base_proc_2.process(proc_1_data)

    result = proc_2_data.loc[start_datetime:end_datetime]
    if remove_empty:
        result = result.loc[:, result.notna().sum() != 0]

    result_path = os.path.join(output_dir, "base.h5")
    result.to_hdf(result_path, key="data")

    return result_path


def get_outlier_removed_data_list(data_dir: str = OUTPUT_DIR) -> List[TrafficData]:
    data_path_list = glob(os.path.join(data_dir, "*.h5"))

    return [TrafficData.import_from_hdf(path) for path in data_path_list]


def remove_outliers(
    data: pd.DataFrame,
    outlier_processors: List[OutlierProcessor] = [
        InSensorZscoreOutlierProcessor(threshold=3.0),
        HourlyInSensorZscoreOutlierProcessor(threshold=3.0),
        MADOutlierProcessor(threshold=3.0),
        TrimmedMeanOutlierProcessor(rate=0.05, threshold=3.0),
        WinsorizedOutlierProcessor(rate=0.05, zscore_threshold=3.0),
    ],
    output_dir: str = OUTPUT_DIR,
):
    output_paths = []
    for processor in outlier_processors:
        logger.info(f"Processing with {processor.name}")
        processed_data = processor.process(data)

        filename = f"{processor.name}.h5"
        filepath = os.path.join(output_dir, filename)

        processed_data.to_hdf(filepath, key="data")
        output_paths.append(filepath)
    
    return output_paths


# ----- Legacy ----- #


def process_outlier(
    start_datetime: Optional[str] = "2023-12-01 00:00:00",
    end_datetime: Optional[str] = "2024-08-31 23:00:00",
):
    metadata = Metadata.import_from_hdf(METADATA_RAW_PATH)
    speed_limit_map = metadata.data.set_index("LINK_ID")["MAX_SPD"].to_dict()
    lane_map = metadata.data.set_index("LINK_ID")["LANES"].to_dict()

    base_processor = [
        RemovingWeirdZeroOutlierProcessor(),
        TrafficCapacityAbsoluteOutlierProcessor(speed_limit_map, lane_map),
    ]
    outlier_filename_1 = "none_simple.h5"

    # 2) InSensorZscoreOutlierProcessor - 센서별 Z-Score
    set_2_processors = [
        RemovingWeirdZeroOutlierProcessor(),
        TrafficCapacityAbsoluteOutlierProcessor(speed_limit_map, lane_map),
        InSensorZscoreOutlierProcessor(threshold=3.0),
    ]
    outlier_filename_2 = "in_sensor_zscore.h5"

    # 3) HourlyInSensorZscoreOutlierProcessor - 시간대별 센서 Z-Score
    set_3_processors = [
        RemovingWeirdZeroOutlierProcessor(),
        TrafficCapacityAbsoluteOutlierProcessor(speed_limit_map, lane_map),
        HourlyInSensorZscoreOutlierProcessor(threshold=3.0),
    ]
    outlier_filename_3 = "hourly_in_sensor_zscore.h5"

    # 4) MADOutlierProcessor - 중앙값 절대편차(MAD) 기반
    set_4_processors = [
        RemovingWeirdZeroOutlierProcessor(),
        TrafficCapacityAbsoluteOutlierProcessor(speed_limit_map, lane_map),
        MADOutlierProcessor(threshold=3.0),
    ]
    outlier_filename_4 = "mad_outlier.h5"

    # 5) TrimmedMeanOutlierProcessor - 절사평균 기반
    set_5_processors = [
        RemovingWeirdZeroOutlierProcessor(),
        TrafficCapacityAbsoluteOutlierProcessor(speed_limit_map, lane_map),
        TrimmedMeanOutlierProcessor(rate=0.05, threshold=3.0),
    ]
    outlier_filename_5 = "trimmed_mean.h5"

    # 6) WinsorizedOutlierProcessor - 윈저화 기반
    set_6_processors = [
        RemovingWeirdZeroOutlierProcessor(),
        TrafficCapacityAbsoluteOutlierProcessor(speed_limit_map, lane_map),
        WinsorizedOutlierProcessor(rate=0.05, zscore_threshold=3.0),
    ]
    outlier_filename_6 = "winsorized.h5"

    outlier_sets = [
        (base_processor, outlier_filename_1),
        (set_2_processors, outlier_filename_2),
        (set_3_processors, outlier_filename_3),
        (set_4_processors, outlier_filename_4),
        (set_5_processors, outlier_filename_5),
        (set_6_processors, outlier_filename_6),
    ]

    # 이상치는 일반적으로 평활화로 처리하는 것을 보임
    # 평활화는 데이터의 불안정하고 불규칙한 특성을 제거해 주기도 하지만 돌발상황과 같은 특별한 정보를 제거해 버릴 수 도 있음.
    # 즉, 평활화는 특별한 목적이 있을 때, 해당 목적에 맞게 알고리즘을 선택하고 적용해야 함.
    # 본 논문에서는 이러한 특별한 정보를 최대한 보존하면서 이상치를 제거하는 데 초점을 맞춤 (추후 데이터셋 제작 시 중요)

    failed_list = {}
    for outlier_set, filename in outlier_sets:
        traffic_data = TrafficData.import_from_hdf(TRAFFIC_RAW_PATH, dtype=float)
        if start_datetime is not None:
            traffic_data.start_time = start_datetime
        if end_datetime is not None:
            traffic_data.end_time = end_datetime

        ori_data = traffic_data.data.copy()
        failed_set = traffic_data.remove_outliers(outlier_set)
        prc_data = traffic_data.data.copy()

        failed_list[filename] = list(failed_set)
        output_path = os.path.join(OUTPUT_DIR, filename)
        traffic_data.to_hdf(output_path)

        logger.info(f"Processed {filename} from {TRAFFIC_RAW_PATH}")
        if start_datetime is not None or end_datetime is not None:
            logger.info(
                f"Time range: From {traffic_data.data.index[0]} to {traffic_data.data.index[-1]}"
            )

    with open(os.path.join(OUTPUT_DIR, "failed_list.yaml"), "w") as f:
        yaml.dump(failed_list, f)
