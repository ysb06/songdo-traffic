import pytest
from metr.subset import MetrSubset
import os
import geopandas as gpd
from pathlib import Path


def test_folder_existence(raw_dir, target_dir, selected_road_path):
    print("Raw Dataset Directory: ", raw_dir)
    assert os.path.exists(raw_dir)
    assert os.path.exists(target_dir)
    assert os.path.exists(selected_road_path)


def test_generation_small_subset(raw_dir, target_dir, selected_road_path):
    subset = MetrSubset(raw_dir)
    target_sensors_raw: gpd.GeoDataFrame = gpd.read_file(selected_road_path)
    target_sensors = target_sensors_raw["sensor_id"].to_list()
    subset.sensor_filter = target_sensors
    subset.metr_imc.start_time = "2023-11-16"
    subset.metr_imc.end_time = "2024-02-29"
    
    # 아웃라이어, 결측치 500미만 제거

    print(subset.metr_imc.data.shape)

    subset.export(target_dir)
    print([f.name for f in Path(target_dir).iterdir() if f.is_file()])

def test_compare_datasets(target_dir, compare):
    pass
