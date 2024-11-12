import os
from typing import Dict, List, Tuple, Union
from pathlib import Path


from metr.components import (
    TrafficData,
    Metadata,
    AdjacencyMatrix,
    DistancesImc,
    IdList,
    SensorLocations,
)
from metr.components.metr_imc.interpolation import Interpolator

# 한 폴더 내에 여러 데이터에 대한 보간을 수행하는 테스트

def test_interpolation(
    traffic_dataset: List[Tuple[str, str, TrafficData]],
    output_root_dir: str,
    output_traffic_filename: str,
    output_missing_filename: str,
    missing_allow_rate: float,
    interpolators: List[Interpolator],
    interpolation_root_dirs: List[str],
):
    for interpolator, root_dir in zip(interpolators, interpolation_root_dirs):
        root_dir = os.path.join(output_root_dir, root_dir)
        for _, name, traffic_data in traffic_dataset:
            traffic_data = interpolate_traffic_data(
                traffic_data, interpolator, missing_allow_rate=missing_allow_rate
            )

            file_dir = os.path.join(root_dir, name)
            os.makedirs(file_dir, exist_ok=True)
            traffic_filepath = os.path.join(file_dir, output_traffic_filename)
            missing_filepath = os.path.join(file_dir, output_missing_filename)

            traffic_data.to_hdf(traffic_filepath)
            missings_data = traffic_data.missings_info
            missings_data.to_hdf(missing_filepath, key="data")

            print(f"Traffic Data: {traffic_data.data.shape}")
            print(f"Missing Data: {missings_data.shape}")
            print(f"Interpolated data saved to {file_dir}")


def interpolate_traffic_data(
    traffic_data: TrafficData,
    interpolator: Interpolator,
    missing_allow_rate: float = 0.1,
):
    data = traffic_data.data
    missing_allow_count = int(data.shape[0] * missing_allow_rate)
    missing_counts = data.isna().sum()
    # Remove columns with too many missing values
    good_columns = missing_counts[missing_counts <= missing_allow_count].index.to_list()
    traffic_data.sensor_filter = good_columns

    traffic_data.interpolate(interpolator)

    return traffic_data


def test_generating_datasets(
    configs: Dict[str, Union[str, int, float]],
    output_root_dir: str,
    interpolation_root_dirs: List[str],
):
    for directory in interpolation_root_dirs:
        root_dir = os.path.join(output_root_dir, directory)
        folder_list = [
            os.path.join(root_dir, folder)
            for folder in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, folder))
        ]
        raw_dir = configs["raw_data_dir"]
        for directory in folder_list:
            traffic_filepath = os.path.join(directory, configs["traffic_data_filename"])
            metadata_filepath = os.path.join(raw_dir, configs["metadata_filename"])
            distances_filepath = os.path.join(raw_dir, configs["distances_filename"])
            sensor_loc_filepath = os.path.join(
                raw_dir, configs["sensor_locations_filename"]
            )

            traffic_data = TrafficData.import_from_hdf(traffic_filepath)
            id_list = IdList(traffic_data.data.columns.to_list())
            metadata = Metadata.import_from_hdf(metadata_filepath)
            metadata.sensor_filter = id_list.data
            distances = DistancesImc.import_from_csv(distances_filepath)
            distances.sensor_filter = id_list.data
            sensor_locations = SensorLocations.import_from_csv(sensor_loc_filepath)
            sensor_locations.sensor_filter = id_list.data
            adj_mx: AdjacencyMatrix = AdjacencyMatrix.import_from_components(
                id_list, distances
            )

            id_list.to_txt(os.path.join(directory, configs["ids_filename"]))
            metadata.to_hdf(os.path.join(directory, configs["metadata_filename"]))
            distances.to_csv(os.path.join(directory, configs["distances_filename"]))
            sensor_locations.to_csv(
                os.path.join(directory, configs["sensor_locations_filename"])
            )
            adj_mx.to_pickle(os.path.join(directory, configs["adj_mx_filename"]))
            print(f"Generated dataset in {directory}")
