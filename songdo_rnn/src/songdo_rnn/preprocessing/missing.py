import glob
import logging
import os
from typing import List

from metr.components import TrafficData
from metr.components.metr_imc.interpolation import (
    Interpolator,
    LinearInterpolator,
    SplineLinearInterpolator,
)
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

OUTLIER_PROCESSED_DIR = "./output/outlier_processed"
OUTPUT_DIR = "./output/all_processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

logger = logging.getLogger(__name__)


def process_missing():
    interpolators: List[Interpolator] = [
        LinearInterpolator(),
        SplineLinearInterpolator(),
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

            missing_counts = traffic_data_interpolated.isna().sum()
            sensors_with_missing = missing_counts[missing_counts > 0]
            sensors_with_no_missing = missing_counts[missing_counts == 0]
            if not sensors_with_missing.empty:
                logger.warning(
                    f"Not Processed Sensor Count: {sensors_with_missing.count()}"
                )

            traffic_data.sensor_filter = sensors_with_no_missing.index.to_list()
            logger.info(f"Final Data: {traffic_data.data.shape}")
            filename = f"{file_basename[:7]}-{interpolator.__class__.__name__.lower().removesuffix('interpolator')}.h5"
            traffic_data.to_hdf(os.path.join(OUTPUT_DIR, filename))
