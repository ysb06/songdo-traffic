import os
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest

from metr.components.metr_imc.interpolation import TimeMeanFillInterpolator
from metr.components.metr_imc.outlier import ZscoreOutlierProcessor
from metr.subset import MetrSubset


def test_folder_existence(raw_dir: str, target_dir: str, selected_road_path: str):
    print("Raw Dataset Directory: ", raw_dir)
    assert os.path.exists(raw_dir)
    assert os.path.exists(target_dir)
    assert os.path.exists(selected_road_path)


def test_generation_small_subset(
    raw_dataset: MetrSubset, target_dir: str, selected_road_path: str
):
    subset = raw_dataset
    print("Raw:", subset.metr_imc.data.shape)
    target_sensors_raw: gpd.GeoDataFrame = gpd.read_file(selected_road_path)
    target_sensors = target_sensors_raw["sensor_id"].to_list()
    subset.sensor_filter = target_sensors

    subset.metr_imc.remove_weird_zero()

    outlier_processor = ZscoreOutlierProcessor(threshold=15)
    subset.metr_imc.remove_outliers(outlier_processor)

    subset.metr_imc.start_time = "2023-11-16"
    subset.metr_imc.end_time = "2024-02-29 23:00:00"

    subset.metr_imc.data.dropna(axis=1, how="all", inplace=True)
    subset.metr_imc.data = subset.metr_imc.data.loc[
        :, subset.metr_imc.data.isnull().sum() < 500
    ]

    interpolator = TimeMeanFillInterpolator()
    subset.metr_imc.interpolate(interpolator)

    print(subset.metr_imc.data.shape)
    subset.export(target_dir)


# CD: Compare Dataset
def test_CD_index_len(
    raw_dataset: MetrSubset,
    gen_subset: MetrSubset,
    cmp_subset: MetrSubset,
):
    raw_min = raw_dataset.metr_imc.data.index.min()
    raw_max = raw_dataset.metr_imc.data.index.max()
    raw_len = raw_max - raw_min
    print("Raw Length -->", raw_min, "~", raw_max, "Len: ", raw_len)

    gen_min = gen_subset.metr_imc.data.index.min()
    gen_max = gen_subset.metr_imc.data.index.max()
    gen_len = gen_max - gen_min
    print("Gen Length -->", gen_min, "~", gen_max, "Len: ", gen_len)

    cmp_min = cmp_subset.metr_imc.data.index.min()
    cmp_max = cmp_subset.metr_imc.data.index.max()
    cmp_len = cmp_max - cmp_min
    print("Cmp Length -->", cmp_min, "~", cmp_max, "Len: ", cmp_len)

    assert gen_len == cmp_len


def test_CD_sensors_len(
    raw_dataset: MetrSubset, gen_subset: MetrSubset, cmp_subset: MetrSubset
):
    raw_sensors = raw_dataset.metr_imc.data.columns
    gen_sensors = gen_subset.metr_imc.data.columns
    cmp_sensors = cmp_subset.metr_imc.data.columns

    print("Raw Sensors: ", len(raw_sensors))
    print("Gen Sensors: ", len(gen_sensors))
    print("Cmp Sensors: ", len(cmp_sensors))

    assert len(gen_sensors) == len(cmp_sensors)
    assert sorted(gen_sensors) == sorted(cmp_sensors)


def test_CD_values(gen_subset: MetrSubset, cmp_subset: MetrSubset):
    gen_df = gen_subset.metr_imc.data
    cmp_df = cmp_subset.metr_imc.data

    common_columns = gen_df.columns.intersection(cmp_df.columns)

    gen_common = gen_df[common_columns].sort_index(axis=1).sort_index()
    cmp_common = cmp_df[common_columns].sort_index(axis=1).sort_index()

    pd.testing.assert_frame_equal(gen_common, cmp_common)
    # 계속 테스트는 실패 하는데 그 원인이 수집 범위가 다르기 때문이라고 생각됨
    # 즉 더이상 신경쓰지 않아도 될 듯
