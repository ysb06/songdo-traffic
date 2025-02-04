import glob
import logging
import os
from typing import List, Optional

from metr.components import TrafficData
from metr.components.metr_imc.interpolation import (
    Interpolator,
    LinearInterpolator,
    SplineLinearInterpolator,
    TimeMeanFillInterpolator,
    ShiftFillInterpolator,
    MonthlyMeanFillInterpolator,
)
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

OUTLIER_PROCESSED_DIR = "./output/outlier_processed"
OUTPUT_DIR = "./output/missing_processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

logger = logging.getLogger(__name__)


def interpolate_missing(
    data_list: List[TrafficData],
    start_datetime: Optional[str] = "2024-01-01 00:00:00",
    end_datetime: Optional[str] = "2024-08-31 23:00:00",
):
    interpolators: List[Interpolator] = [
        LinearInterpolator(),
        # SplineLinearInterpolator(),
        TimeMeanFillInterpolator(),
        # ShiftFillInterpolator(periods=7),
        MonthlyMeanFillInterpolator(),
    ]

    for raw_data in data_list:
        data = raw_data.data
        data_basename = os.path.basename(raw_data.path if raw_data.path else "unknown")
        data_basename = data_basename.removesuffix(".h5")

        for interpolator in interpolators:
            interpolator_name = interpolator.__class__.__name__.lower()
            interpolator_name = interpolator_name.removesuffix("interpolator")
            logger.info(f'Interpolating with "{interpolator_name}"')

            interpolated_data = interpolator.interpolate(data)
            interpolated_data = interpolated_data.loc[start_datetime:end_datetime]

            filepath = os.path.join(OUTPUT_DIR, f"{data_basename}-{interpolator_name}.h5")
            interpolated_data.to_hdf(filepath, key="data")
            logger.info(f"Interpolated data saved to {filepath}")


def process_missing(
    start_datetime: Optional[str] = "2024-01-01 00:00:00",
    end_datetime: Optional[str] = "2024-08-31 23:00:00",
):
    interpolators: List[Interpolator] = [
        LinearInterpolator(),
        SplineLinearInterpolator(),
        TimeMeanFillInterpolator(),
        # ShiftFillInterpolator(periods=7),
        MonthlyMeanFillInterpolator(),
    ]
    traffic_files = glob.glob(os.path.join(OUTLIER_PROCESSED_DIR, "*.h5"))
    for file_path in traffic_files:
        for interpolator in interpolators:
            file_basename = os.path.basename(file_path)
            logger.info(
                f"Processing {file_basename} with {interpolator.__class__.__name__}"
            )
            traffic_data = TrafficData.import_from_hdf(file_path)
            logger.info(f"Original Data: {traffic_data.data.shape}")

            traffic_data_original = traffic_data.data.copy()
            traffic_data.interpolate(interpolator)
            traffic_data_interpolated = traffic_data.data.copy()
            # 1610001700
            missing_counts = traffic_data_interpolated.isna().sum()
            sensors_with_missing = missing_counts[missing_counts > 0]
            sensors_with_no_missing = missing_counts[missing_counts == 0]
            if not sensors_with_missing.empty:
                logger.warning(
                    f"Not Processed Sensor Count: {sensors_with_missing.count()}"
                )

            traffic_data.sensor_filter = sensors_with_no_missing.index.to_list()
            traffic_data.start_time = start_datetime
            traffic_data.end_time = end_datetime
            logger.info(f"Final Data: {traffic_data.data.shape}")

            outlier_basename = file_basename.split(".")[0]
            interpolator_basename = (
                interpolator.__class__.__name__.lower().removesuffix("interpolator")
            )
            filename = f"{outlier_basename}-{interpolator_basename}.h5"
            traffic_data.to_hdf(os.path.join(OUTPUT_DIR, filename))
