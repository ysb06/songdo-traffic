import logging
import os
from glob import glob
from typing import List, Optional, Tuple

import pandas as pd
from metr.components.metr_imc.outlier import (
    OutlierProcessor,
    RemovingWeirdZeroOutlierProcessor,
    TrafficCapacityAbsoluteOutlierProcessor,
)
from metr.components import TrafficData

logger = logging.getLogger(__name__)


def generate_base_data(
    training_raw: pd.DataFrame,
    metadata: pd.DataFrame,
    traffic_capacity_adjustment_rate: float = 2.0,
    output_dir: Optional[str] = None,
) -> pd.DataFrame:
    speed_limit_map = metadata.set_index("LINK_ID")["MAX_SPD"].to_dict()
    lane_map = metadata.set_index("LINK_ID")["LANES"].to_dict()
    base_processor: List[OutlierProcessor] = [
        RemovingWeirdZeroOutlierProcessor(),
        TrafficCapacityAbsoluteOutlierProcessor(
            speed_limit_map,
            lane_map,
            adjustment_rate=traffic_capacity_adjustment_rate,
        ),
    ]

    raw_copy = training_raw.copy()
    for processor in base_processor:
        raw_copy = processor.process(raw_copy)

    if output_dir is not None:
        filepath = os.path.join(output_dir, "base.h5")
        raw_copy.to_hdf(filepath, key="data")

    return raw_copy


def remove_outliers(
    raw: pd.DataFrame,
    outlier_processors: List[OutlierProcessor],
    output_dir: Optional[str] = None,
) -> List[Tuple[pd.DataFrame, str]]:
    results: List[Tuple[pd.DataFrame, str]] = []
    for processor in outlier_processors:
        result_data = processor.process(raw)
        results.append((result_data, processor.name))

        if output_dir is not None:
            filepath = os.path.join(output_dir, f"{processor.name}.h5")
            result_data.to_hdf(filepath, key="data")

    return results


def load_outlier_removed_data(data_dir: str) -> List[Tuple[pd.DataFrame, str]]:
    glob_pattern = os.path.join(data_dir, "*.h5")
    data_paths = glob(glob_pattern)
    data_list: List[Tuple[pd.DataFrame, str]] = []
    for path in data_paths:
        raw = TrafficData.import_from_hdf(path)
        data_list.append((raw.data, os.path.basename(path).removesuffix(".h5")))

    return data_list
