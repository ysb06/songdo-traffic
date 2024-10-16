# tests/outliers/conftest.py
import pytest
import yaml
import os
from dataclasses import dataclass

from metr.components.metadata import Metadata
from metr.components.metr_imc.traffic_data import TrafficData


@dataclass
class Configs:
    raw_dir: str
    selected_node_path: str
    misc_dir: str
    out_root_dir: str
    outlier_out_dir: str
    # filenames
    traffic_data_filename: str
    traffic_training_data_filename: str
    traffic_test_data_filename: str
    metadata_filename: str
    adj_mx_filename: str
    distances_filename: str
    ids_filename: str
    sensor_locations_filename: str


def load_configs():
    yaml_path = os.path.join(os.path.dirname(__file__), "config.yaml")

    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)

    config = Configs(**config)

    # Mkdirs
    os.makedirs(config.out_root_dir, exist_ok=True)
    os.makedirs(config.outlier_out_dir, exist_ok=True)

    return config


raw_config = load_configs()


@pytest.fixture
def configs():
    return raw_config

@pytest.fixture
def selected_training_traffic_data():
    file_path = os.path.join(raw_config.out_root_dir, raw_config.traffic_training_data_filename)
    return TrafficData.import_from_hdf(file_path)

@pytest.fixture
def road_metadata():
    file_path = os.path.join(raw_config.raw_dir, raw_config.metadata_filename)
    return Metadata.import_from_hdf(file_path)


@pytest.fixture
def outlier_output_dir():
    return raw_config.outlier_out_dir

@pytest.fixture
def outlier_output_path():
    paths = {
        "simple_absolute": os.path.join(raw_config.outlier_out_dir, "metr-imc-01-absolute-simple.h5"),
        "traffic_capacity_absolute": os.path.join(raw_config.outlier_out_dir, "metr-imc-02-absolute-traffic-capacity.h5"),
        "simple_zscore": os.path.join(raw_config.outlier_out_dir, "metr-imc-03-zscore-simple.h5"),
        "hourly_zscore": os.path.join(raw_config.outlier_out_dir, "metr-imc-04-zscore-hourly-all.h5"),
        "hourly_in_sensor_zscore": os.path.join(raw_config.outlier_out_dir, "metr-imc-05-zscore-hourly-in-sensor.h5"),
    }

    return paths


