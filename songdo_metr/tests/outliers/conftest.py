# tests/outliers/conftest.py
import pytest
import yaml
import os
from dataclasses import dataclass

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


@pytest.fixture
def configs():
    yaml_path = os.path.join(os.path.dirname(__file__), "config.yaml")

    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)

    config = Configs(**config)

    # Mkdirs
    os.makedirs(config.out_root_dir, exist_ok=True)
    os.makedirs(config.outlier_out_dir, exist_ok=True)

    return config
