import logging
import os
import glob

import geopandas as gpd

from metr.components import (
    AdjacencyMatrix,
    DistancesImc,
    IdList,
    Metadata,
    SensorLocations,
    TrafficData,
)
from metr.components.metr_imc.interpolation import (
    TimeMeanFillInterpolator,
    LinearInterpolator,
    SplineLinearInterpolator,
    SplineInterpolator,
)
from metr.processors import *
from metr.dataset import generate_file_set

logger = logging.getLogger(__name__)


def run_process():
    os.makedirs(INTERPOLATION_PROCESSED_DIR, exist_ok=True)

    do_all_interpolation()


def do_all_interpolation():
    traffic_files = glob.glob(os.path.join(OUTLIER_PROCESSED_DIR, "*"))
    for traffic_file in traffic_files:
        output_dir_name = os.path.basename(traffic_file).split(".")[0]
        output_dir = os.path.join(INTERPOLATION_PROCESSED_DIR, output_dir_name)
        interpolate(traffic_file, output_dir)


def interpolate(
    traffic_data_path: str,
    output_dir: str,
    metadata_filename: str = METADATA_FILENAME,
    sensor_locations_filename: str = SENSOR_LOCATIONS_FILENAME,
    distances_filename: str = DISTANCES_FILENAME,
    adj_mx_filename: str = ADJ_MX_FILENAME,
):
    traffic_data = TrafficData.import_from_hdf(traffic_data_path, dtype=float)

    processor_set = [
        (LinearInterpolator(), "linear"),  # Linear Interpolation
        (SplineLinearInterpolator(), "spline_linear"),  # Spline Linear Interpolation??
        (SplineInterpolator(), "spline"),  # 3rd Order Spline Interpolation
        (TimeMeanFillInterpolator(), "time_mean_fill"),  # Time Mean Fill Interpolation
    ]

    for processor, name in processor_set:
        traffic_data_path = os.path.join(output_dir, name, TRAFFIC_FILENAME)
        missing_data_path = os.path.join(output_dir, name, MISSING_FILENAME)

        metadata_path = os.path.join(output_dir, name, metadata_filename)
        sensor_loc_path = os.path.join(output_dir, name, sensor_locations_filename)
        distances_path = os.path.join(output_dir, name, distances_filename)
        adj_mx_path = os.path.join(output_dir, name, adj_mx_filename)

        os.makedirs(os.path.join(output_dir, name), exist_ok=True)

        traffic_data.reset_data()
        is_interpolated = traffic_data._raw.isna()
        traffic_data.interpolate(processor)
        traffic_data.to_hdf(traffic_data_path)
        is_interpolated.to_hdf(missing_data_path, key="data")

        generate_file_set(
            traffic_data=traffic_data,
            nodelink_dir=NODELINK_TARGET_DIR,
            road_data_filename=NODELINK_LINK_FILENAME,
            turn_data_filename=NODELINK_TURN_FILENAME,
            metadata_path=metadata_path,
            sensor_locations_path=sensor_loc_path,
            distances_path=distances_path,
            adj_mx_path=adj_mx_path,
        )
        # Todo: 나중에 Generation 과정을 분리하는 것을 고려
