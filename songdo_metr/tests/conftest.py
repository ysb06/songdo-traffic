import pytest

@pytest.fixture
def raw_dir():
    return "../datasets/metr-imc"

@pytest.fixture
def target_dir():
    return "../datasets/metr-imc-subsets"

@pytest.fixture
def selected_road_path():
    return "../datasets/metr-imc-subsets/selected_road.shp"

@pytest.fixture
def comparison_target():
    return "../datasets/metr-imc-subsets/selected_road.shp"


