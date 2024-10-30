import pytest
import yaml
import os
from songdo_arima.utils import get_config
from metr.components import TrafficData

yaml_path = os.path.join(os.path.dirname(__file__), "..", "configs", "config.yaml")
config_raw = get_config(yaml_path)


@pytest.fixture
def configs():
    return config_raw

@pytest.fixture
def traffic_training_data_raw():
    return TrafficData.import_from_hdf(config_raw.traffic_training_data_path)

@pytest.fixture
def traffic_test_data_raw():
    return TrafficData.import_from_hdf(config_raw.traffic_test_data_path)