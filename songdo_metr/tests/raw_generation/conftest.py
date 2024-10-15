# tests/outliers/conftest.py
import pytest
import yaml
import os
from dataclasses import dataclass


@dataclass
class Configs:
    NODELINK_DATA_URL: str
    NODELINK_TARGET_DIR: str
    IMCRTS_TARGET_DIR: str
    target_dir: str
    target_start_date: str
    target_end_date: str
    training_start_datetime: str
    training_end_datetime: str
    test_start_datetime: str
    test_end_datetime: str


@pytest.fixture
def configs():
    yaml_path = os.path.join(os.path.dirname(__file__), "config.yaml")

    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)

    config = Configs(**config)
    return config
