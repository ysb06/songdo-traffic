import os
from .conftest import Configs
from metr.components.metr_imc import TrafficData


def test_training_data_split(configs: Configs):
    raw_data_path = os.path.join(configs.target_dir, "metr-imc.h5")
    target_path = os.path.join(configs.target_dir, "metr-imc-training.h5")

    traffic_data = TrafficData.import_from_hdf(raw_data_path)
    traffic_data.start_time = configs.training_start_datetime
    traffic_data.end_time = configs.training_end_datetime

    traffic_data.to_hdf(target_path)


def test_test_data_split(configs: Configs):
    raw_data_path = os.path.join(configs.target_dir, "metr-imc.h5")
    target_path = os.path.join(configs.target_dir, "metr-imc-test.h5")

    traffic_data = TrafficData.import_from_hdf(raw_data_path)
    traffic_data.start_time = configs.test_start_datetime
    traffic_data.end_time = configs.test_end_datetime

    traffic_data.to_hdf(target_path)
