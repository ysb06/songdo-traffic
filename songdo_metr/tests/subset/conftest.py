import os
import pytest
import geopandas as gpd

from metr.subset import MetrSubset

RAW_DIR = "../datasets/metr-imc"
TARGET_DIR = "../datasets/metr-imc-subsets"
COMPARISON_TARGET_DIR = "../datasets/metr-imc_legacy_2/subsets/metr-4-combined"
SELECTED_ROAD_PATH = "../datasets/metr-imc-subsets/selected_road.shp"
NODELINK_DIR = os.path.join(RAW_DIR, "nodelink")
LINK_DATA_PATH = os.path.join(NODELINK_DIR, "imc_link.shp")


@pytest.fixture
def raw_dir():
    return RAW_DIR


@pytest.fixture
def target_dir():
    return TARGET_DIR


@pytest.fixture
def selected_road_path():
    return SELECTED_ROAD_PATH


@pytest.fixture
def comparison_target_dir():
    return COMPARISON_TARGET_DIR

@pytest.fixture
def nodelink_dir():
    return NODELINK_DIR

@pytest.fixture
def nodelink_road_data_path():
    return LINK_DATA_PATH


@pytest.fixture
def nodelink_road_data():
    return gpd.read_file(LINK_DATA_PATH)


@pytest.fixture
def raw_dataset():
    return MetrSubset(RAW_DIR)


@pytest.fixture
def gen_subset():
    return MetrSubset(TARGET_DIR)


@pytest.fixture
def cmp_subset():
    return MetrSubset(
        COMPARISON_TARGET_DIR, distances_imc_filename="distances_imc_2023.csv"
    )
