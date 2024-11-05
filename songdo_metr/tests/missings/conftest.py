import glob
import os
from pathlib import Path
from typing import Dict, List, Tuple, Union

import pytest
import yaml

from metr.components import TrafficData
from metr.components.metr_imc.interpolation import TimeMeanFillInterpolator


def load_configs():
    yaml_path = os.path.join(os.path.dirname(__file__), "config.yaml")

    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)

    return config


raw_config: Dict[str, Union[str, int, float]] = load_configs()


@pytest.fixture
def configs():
    return raw_config


@pytest.fixture
def traffic_dataset():
    target_dir = raw_config["loading_target_dir"]
    files = glob.glob(os.path.join(target_dir, "*.h5"))
    files.sort()

    result: List[Tuple[str, str, TrafficData]] = []
    for file_path in files:
        path = file_path
        name = Path(os.path.basename(file_path)).stem
        data = TrafficData.import_from_hdf(file_path)

        result.append((path, name, data))

    return result


@pytest.fixture
def output_root_dir():
    return raw_config["output_dir"]


@pytest.fixture
def missing_allow_rate():
    return raw_config["missing_allow_rate"]


@pytest.fixture
def output_traffic_filename():
    return raw_config["traffic_data_filename"]


@pytest.fixture
def output_missing_filename():
    return raw_config["traffic_missing_data_filename"]

@pytest.fixture
def interpolators():
    return [TimeMeanFillInterpolator()]

@pytest.fixture
def interpolation_root_dirs():
    return ["time_mean_avg"]
