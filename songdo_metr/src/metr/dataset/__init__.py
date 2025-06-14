# Todo: 차후에는 raw_data로 패키지 명을 변경하기
from typing import Tuple

from metr.components import (AdjacencyMatrix, DistancesImc, IdList, Metadata,
                             SensorLocations, TrafficData)


def generate_file_set(
    traffic_data: TrafficData,
    nodelink_dir: str,
    road_data_filename: str,
    turn_data_filename: str,
    metadata_path: str,
    sensor_locations_path: str,
    distances_path: str,
    adj_mx_path: str,
):
    metadata, sensor_locations, distances, adj_mx = generate_dataset(
        traffic_data, nodelink_dir, road_data_filename, turn_data_filename
    )

    metadata.to_hdf(metadata_path)
    sensor_locations.to_csv(sensor_locations_path)
    distances.to_csv(distances_path)
    adj_mx.to_pickle(adj_mx_path)

    return metadata, sensor_locations, distances, adj_mx


def generate_dataset(
    traffic_data: TrafficData,
    nodelink_dir: str,
    road_data_filename: str,
    turn_data_filename: str,
) -> Tuple[Metadata, SensorLocations, DistancesImc, AdjacencyMatrix]:
    # Sensor IDs
    metr_ids = IdList(traffic_data.data.columns.to_list())
    
    # Metadata
    metadata = Metadata.import_from_nodelink(
        nodelink_dir, road_filename=road_data_filename
    )
    metadata.sensor_filter = metr_ids.data

    # Sensor Locations
    sensor_locations = SensorLocations.import_from_nodelink(
        nodelink_dir, road_filename=road_data_filename
    )
    sensor_locations.sensor_filter = metr_ids.data

    # Distances
    distances = DistancesImc.import_from_nodelink(
        nodelink_dir,
        road_filename=road_data_filename,
        turn_filename=turn_data_filename,
        target_ids=metr_ids.data,
        distance_limit=9000,
    )

    # Adjacency Matrix
    adj_mx: AdjacencyMatrix = AdjacencyMatrix.import_from_components(
        metr_ids, distances
    )

    return metadata, sensor_locations, distances, adj_mx
