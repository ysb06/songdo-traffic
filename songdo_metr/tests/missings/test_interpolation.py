import glob
import os
from typing import List
from pathlib import Path


from metr.components import (
    TrafficData,
    Metadata,
    AdjacencyMatrix,
    DistancesImc,
    IdList,
    SensorLocations,
)
from metr.components.metr_imc.interpolation import TimeMeanFillInterpolator
from tests.missings.conftest import Configs

print("test_interpolation started")


def test_time_mean_interpolation(
    traffic_data_list: List[TrafficData],
    traffic_data_filename_list: List[str],
    output_dir: str,
    configs: Configs,
):
    target_filename = configs.traffic_data_filename

    for traffic_data, tr_data_filename in zip(traffic_data_list, traffic_data_filename_list):
        data = traffic_data.data
        missing_allow_count = int(data.shape[0] * configs.missing_allow_rate)
        print(target_filename, ":")
        missing_counts = data.isna().sum()
        good_columns = missing_counts[missing_counts <= missing_allow_count].index
        good_data = data[good_columns]
        print(data.shape, "->", good_data.shape)

        traffic_data.data = good_data
        traffic_data.interpolate(TimeMeanFillInterpolator())
        file_dir = os.path.join(output_dir, "time_mean_avg",Path(tr_data_filename).stem)
        filepath = os.path.join(file_dir, target_filename)
        os.makedirs(file_dir, exist_ok=True)
        traffic_data.to_hdf(filepath)
        print(f"Interpolated data saved to {filepath}")


def test_finishing(
    output_dir: str,
    configs: Configs,
):
    folder_list = [
        folder
        for folder in os.listdir(output_dir)
        if os.path.isdir(os.path.join(output_dir, folder))
    ]
    folder_paths = [os.path.join(output_dir, folder) for folder in folder_list]
    raw_data_dir = configs.raw_data_dir

    for foler_path in folder_paths:
        traffic_filepath = os.path.join(foler_path, configs.traffic_data_filename)
        metadata_filepath = os.path.join(raw_data_dir, configs.metadata_filename)
        distances_filepath = os.path.join(raw_data_dir, configs.distances_filename)
        sensor_locations_filepath = os.path.join(
            raw_data_dir, configs.sensor_locations_filename
        )

        traffic_data = TrafficData.import_from_hdf(traffic_filepath)
        id_list = IdList(traffic_data.data.columns.to_list())
        metadata = Metadata.import_from_hdf(metadata_filepath)
        metadata.sensor_filter = id_list.data
        distances = DistancesImc.import_from_csv(distances_filepath)
        distances.sensor_filter = id_list.data
        sensor_locations = SensorLocations.import_from_csv(sensor_locations_filepath)
        sensor_locations.sensor_filter = id_list.data
        adj_mx: AdjacencyMatrix = AdjacencyMatrix.import_from_components(id_list, distances)

        id_list.to_txt(os.path.join(foler_path, configs.ids_filename))
        metadata.to_hdf(os.path.join(foler_path, configs.metadata_filename))
        distances.to_csv(os.path.join(foler_path, configs.distances_filename))
        sensor_locations.to_csv(
            os.path.join(foler_path, configs.sensor_locations_filename)
        )
        adj_mx.to_pickle(os.path.join(foler_path, configs.adj_mx_filename))
