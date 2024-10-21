import os
import pytest
import yaml
from metr.components import TrafficData
import glob
from typing import List
from dataclasses import dataclass


@dataclass
class Configs:
    raw_data_dir: str
    loading_target_dir: str
    output_dir: str
    missing_allow_rate: float
    # filenames
    traffic_data_filename: str
    metadata_filename: str
    adj_mx_filename: str
    distances_filename: str
    ids_filename: str
    sensor_locations_filename: str


def load_configs():
    yaml_path = os.path.join(os.path.dirname(__file__), "config.yaml")

    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)

    print(config)
    config = Configs(**config)

    return config


raw_config = load_configs()


@pytest.fixture
def configs():
    return raw_config


@pytest.fixture
def traffic_data_list():
    target_dir = raw_config.loading_target_dir
    files = glob.glob(os.path.join(target_dir, "*.h5"))
    files.sort()
    print("Files:", files)
    result = []
    for file_path in files:
        result.append(TrafficData.import_from_hdf(file_path))

    return result


@pytest.fixture
def traffic_data_filename_list():
    target_dir = raw_config.loading_target_dir
    files = glob.glob(os.path.join(target_dir, "*.h5"))
    files.sort()
    print("Names:", files)
    result = []
    for file_path in files:
        result.append(os.path.basename(file_path))

    return result


@pytest.fixture
def output_dir():
    return raw_config.output_dir

@pytest.fixture
def missing_allow_rate():
    return raw_config.missing_allow_rate
