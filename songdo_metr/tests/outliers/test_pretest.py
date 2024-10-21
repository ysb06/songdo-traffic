import os

from tests.outliers.conftest import Configs
from metr.components import (
    TrafficData,
    Metadata,
    IdList,
    SensorLocations,
    AdjacencyMatrix,
    DistancesImc,
)
import geopandas as gpd
from folium.folium import Map
import pytest


@pytest.mark.run(order=1)
def test_path(configs: Configs):
    raw_traffic_data_path = os.path.join(configs.raw_dir, configs.traffic_data_filename)
    raw_metadata_path = os.path.join(configs.raw_dir, configs.metadata_filename)
    raw_ids_path = os.path.join(configs.raw_dir, configs.ids_filename)
    raw_distances_path = os.path.join(configs.raw_dir, configs.distances_filename)
    raw_adj_mx_path = os.path.join(configs.raw_dir, configs.adj_mx_filename)
    raw_sensor_locations_path = os.path.join(
        configs.raw_dir, configs.sensor_locations_filename
    )

    assert os.path.exists(raw_traffic_data_path)
    assert os.path.exists(raw_metadata_path)
    assert os.path.exists(raw_ids_path)
    assert os.path.exists(raw_distances_path)
    assert os.path.exists(raw_adj_mx_path)
    assert os.path.exists(raw_sensor_locations_path)


@pytest.mark.run(order=2)
def test_import(configs: Configs):
    raw_traffic_data_path = os.path.join(configs.raw_dir, configs.traffic_data_filename)
    raw_metadata_path = os.path.join(configs.raw_dir, configs.metadata_filename)

    map_output_path = os.path.join(configs.out_root_dir, "selected_nodes.html")

    TrafficData.import_from_hdf(raw_traffic_data_path)
    Metadata.import_from_hdf(raw_metadata_path)
    selected_nodes: gpd.GeoDataFrame = gpd.read_file(configs.selected_node_path)
    map: Map = selected_nodes.explore()
    map.save(map_output_path)


@pytest.mark.run(order=3)
def test_selected_subsets_generation(configs: Configs):
    raw_traffic_data_path = os.path.join(
        configs.raw_dir, configs.traffic_training_data_filename
    )
    raw_metadata_path = os.path.join(configs.raw_dir, configs.metadata_filename)
    raw_sensor_locations_path = os.path.join(
        configs.raw_dir, configs.sensor_locations_filename
    )
    raw_distances_path = os.path.join(configs.raw_dir, configs.distances_filename)

    metr_imc = TrafficData.import_from_hdf(raw_traffic_data_path)
    metr_imc_meta = Metadata.import_from_hdf(raw_metadata_path)
    sensor_locs = SensorLocations.import_from_csv(raw_sensor_locations_path)
    dist_imc = DistancesImc.import_from_csv(raw_distances_path)

    selected_nodes = gpd.read_file(configs.selected_node_path)
    metr_ids = IdList(selected_nodes["LINK_ID"].values)

    metr_imc.sensor_filter = metr_ids.data
    metr_imc_meta.sensor_filter = metr_ids.data
    sensor_locs.sensor_filter = metr_ids.data
    dist_imc.sensor_filter = metr_ids.data
    adj_mx = AdjacencyMatrix.import_from_components(metr_ids, dist_imc)

    selected_traffic_data_path = os.path.join(
        configs.out_root_dir, configs.traffic_training_data_filename
    )
    selected_metadata_path = os.path.join(
        configs.out_root_dir, configs.metadata_filename
    )
    selected_metr_ids_path = os.path.join(configs.out_root_dir, configs.ids_filename)
    selected_sensor_locations_path = os.path.join(
        configs.out_root_dir, configs.sensor_locations_filename
    )
    selected_distances_path = os.path.join(
        configs.out_root_dir, configs.distances_filename
    )
    selected_adj_mx_path = os.path.join(configs.out_root_dir, configs.adj_mx_filename)

    metr_imc.to_hdf(selected_traffic_data_path)
    metr_imc_meta.to_hdf(selected_metadata_path)
    metr_ids.to_txt(selected_metr_ids_path)
    sensor_locs.to_csv(selected_sensor_locations_path)
    dist_imc.to_csv(selected_distances_path)
    adj_mx.to_pickle(selected_adj_mx_path)


@pytest.mark.run(order=4)
def test_test_subset_generation(configs: Configs):
    raw_traffic_data_path = os.path.join(
        configs.raw_dir, configs.traffic_test_data_filename
    )
    test_traffic_data_path = os.path.join(
        configs.out_root_dir, configs.traffic_test_data_filename
    )

    metr_imc = TrafficData.import_from_hdf(raw_traffic_data_path)
    metr_imc.to_hdf(test_traffic_data_path)

    # 이것은 테스트의 raw파일이고 실제 테스트를 위해서는 생성된 Subset의 교집합으로 별도의 테스트셋을 생성해야함
