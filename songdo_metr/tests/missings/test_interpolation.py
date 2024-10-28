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
    target_missing_filename = configs.traffic_missing_data_filename

    for traffic_data, tr_data_filename in zip(traffic_data_list, traffic_data_filename_list):
        data = traffic_data.data
        missing_allow_count = int(data.shape[0] * configs.missing_allow_rate)
        print(target_filename, ":")
        missing_counts = data.isna().sum()
        # Remove columns with too many missing values
        good_columns = missing_counts[missing_counts <= missing_allow_count].index.to_list()
        traffic_data.sensor_filter = good_columns
        print(traffic_data._raw.shape, "->", traffic_data.data.shape)

        traffic_data.interpolate(TimeMeanFillInterpolator())
        file_dir = os.path.join(output_dir, "time_mean_avg", Path(tr_data_filename).stem)
        traffic_filepath = os.path.join(file_dir, target_filename)
        missing_filepath = os.path.join(file_dir, target_missing_filename)
        os.makedirs(file_dir, exist_ok=True)

        traffic_data.to_hdf(traffic_filepath)
        missings_data = traffic_data.missings_info
        missings_data.to_hdf(missing_filepath, key="data")

        print(f"Traffic Data: {traffic_data.data.shape}")
        print(f"Missing Data: {missings_data.shape}")
        print(f"Interpolated data saved to {file_dir}")


def test_generating_full_dataset(
    output_dir: str,
    configs: Configs,
):
    output_dir = os.path.join(output_dir, "time_mean_avg")
    folder_list = [
        folder
        for folder in os.listdir(output_dir)
        if os.path.isdir(os.path.join(output_dir, folder))
    ]
    folder_paths = [os.path.join(output_dir, folder) for folder in folder_list]
    raw_data_dir = configs.raw_data_dir

    for folder_path in folder_paths:
        traffic_filepath = os.path.join(folder_path, configs.traffic_data_filename)
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

        id_list.to_txt(os.path.join(folder_path, configs.ids_filename))
        metadata.to_hdf(os.path.join(folder_path, configs.metadata_filename))
        distances.to_csv(os.path.join(folder_path, configs.distances_filename))
        sensor_locations.to_csv(
            os.path.join(folder_path, configs.sensor_locations_filename)
        )
        adj_mx.to_pickle(os.path.join(folder_path, configs.adj_mx_filename))
