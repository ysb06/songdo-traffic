import os
from typing import List, Optional
from glob import glob
import logging

import pandas as pd
from matplotlib import pyplot as plt
from metr.components.metr_imc import TrafficData
import pytest
from songdo_rnn.plot import plot_missing

logger = logging.getLogger(__name__)

TRAFFIC_RAW_PATH = "../datasets/metr-imc/metr-imc.h5"
BASE_OUTLIER_PATH = "./output/outlier_processed/base_outlier.h5"
HISZ_OUTLIER_PATH = "./output/outlier_processed/hourlyinsensorzscore.h5"
INTERPOLATION_DIR = "./output/missing_processed"


@pytest.fixture
def max_sensors():
    return None


@pytest.fixture
def target_sensors():
    return ["1610029200"]


def test_base_outlier_processor_visualization(
    max_sensors: int, target_sensors: List[str]
):
    TRAFFIC_RAW_PATH = "../datasets/metr-imc/metr-imc.h5"
    BASE_OUTLIER_PATH = "./output/outlier_processed/base_outlier.h5"

    raw = TrafficData.import_from_hdf(TRAFFIC_RAW_PATH).data
    base_outlier_data = TrafficData.import_from_hdf(BASE_OUTLIER_PATH).data

    visualize(
        raw,
        base_outlier_data,
        title="Base Processor",
        max_sensors=max_sensors,
        target_sensors=target_sensors,
    )


def test_hourlyinsensorzscore_outlier_processor_visualization(
    max_sensors: int, target_sensors: List[str]
):
    TRAFFIC_RAW_PATH = "../datasets/metr-imc/metr-imc.h5"
    HISZ_OUTLIER_PATH = "./output/outlier_processed/hourlyinsensorzscore.h5"

    raw = TrafficData.import_from_hdf(TRAFFIC_RAW_PATH).data
    hisz_outlier_data = TrafficData.import_from_hdf(HISZ_OUTLIER_PATH).data

    visualize(
        raw,
        hisz_outlier_data,
        title="Hourly In Sensor Z-score Processor",
        max_sensors=max_sensors,
        target_sensors=target_sensors,
    )


def test_base_missing_interpolation_visualization(
    max_sensors: int, target_sensors: List[str]
):
    data_path_list = glob(os.path.join(INTERPOLATION_DIR, "*.h5"))
    raw = TrafficData.import_from_hdf(BASE_OUTLIER_PATH).data

    for path in data_path_list:
        data = TrafficData.import_from_hdf(path).data
        name = os.path.basename(path).removesuffix(".h5")

        print(f"{name} NaN Count: {data.isna().sum().sum()}")
        print(f"{name} NaN Columns: {data.isna().any(axis=0).sum()} / {data.shape[1]}")

        visualize(
            raw,
            data,
            title=name,
            max_sensors=max_sensors,
            target_sensors=target_sensors,
        )


def visualize(
    original_df: pd.DataFrame,
    processed_df: pd.DataFrame,
    title: Optional[str] = None,
    target_sensors: Optional[List[str]] = None,
    max_sensors: Optional[int] = None,
):
    sensors = original_df.columns
    if target_sensors is not None:
        sensors = target_sensors
    if max_sensors is not None and max_sensors < len(sensors):
        sensors = sensors[:max_sensors]

    intersected_index = original_df.index.intersection(processed_df.index)
    for sensor in sensors:
        s_orig = original_df[sensor].loc[intersected_index]
        s_proc = processed_df[sensor].loc[intersected_index]

        print(f"Sensor: {sensor} | NaN Count: {s_proc.isna().sum()}")
        plot_missing(s_orig, s_proc, title)
