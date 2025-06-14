import os
from metr.processors.raw import (
    RAW_MISCELLANEOUS_DIR_PATH,
    RAW_NODELINK_DIR_PATH,
    RAW_IMCRTS_DIR_PATH,
    RAW_NODELINK_LINK_PATH,
    RAW_NODELINK_TURN_PATH,
    RAW_IMCRTS_PATH,
    RAW_METR_IMC_PATH,
)
from metr.components import TrafficData
from datetime import datetime
import pandas as pd


def test_file_existance():
    assert os.path.exists(RAW_MISCELLANEOUS_DIR_PATH)
    assert os.path.exists(RAW_NODELINK_DIR_PATH)
    assert os.path.exists(RAW_IMCRTS_DIR_PATH)
    assert os.path.exists(RAW_NODELINK_LINK_PATH)
    assert os.path.exists(RAW_NODELINK_TURN_PATH)
    assert os.path.exists(RAW_IMCRTS_PATH)
    assert os.path.exists(RAW_METR_IMC_PATH)


def test_check_raw_info():
    tr_data = TrafficData.import_from_hdf(RAW_METR_IMC_PATH)
    print(f"Raw Shape: {tr_data.data.shape}")
    assert tr_data.data.shape[0] > 0, "Traffic data should not be empty"
    assert tr_data.data.shape[1] > 0, "Traffic data should have columns(sensors)"
    print(f"Index Range: {tr_data.data.index.min()} - {tr_data.data.index.max()}")
    print(f"Index Count: {len(tr_data.data.index)}")
    print(tr_data.data)


def test_test_range_inspection():
    data = TrafficData.import_from_hdf(RAW_METR_IMC_PATH).data

    start_time = datetime(2025, 1, 1, 0)
    end_time = datetime(2025, 3, 9, 23)

    test_data = data.loc[start_time:end_time]

    total_columns = len(test_data.columns)
    non_nan_columns = test_data.columns[~test_data.isna().any()]
    nan_columns = total_columns - len(non_nan_columns)
    print(f"전체 열 수: {total_columns}, NaN이 없는 열 개수: {len(non_nan_columns)}")
    print(f"NaN이 있는 열 수: {nan_columns}")


def test_training_data():
    data = TrafficData.import_from_hdf(RAW_METR_IMC_PATH).data

    training_start_time = datetime(2022, 11, 1, 0)
    training_end_time = datetime(2023, 12, 31, 23)
    test_start_time = datetime(2024, 1, 1, 0)
    test_end_time = datetime(2024, 3, 9, 23)

    training_data = data.loc[training_start_time:training_end_time]
    test_data = data.loc[test_start_time:test_end_time]
    non_nan_columns = test_data.columns[~test_data.isna().any()]
    selected_training_data = training_data[non_nan_columns]

    print(f"Training Data Shape: {training_data.shape}")
    print_nan_distribution(training_data)
    print(f"Selected Data Shape: {selected_training_data.shape}")
    print_nan_distribution(selected_training_data)


def print_nan_distribution(data: pd.DataFrame):
    # 각 열의 NaN 비율 계산
    nan_percentages = data.isna().mean() * 100

    # 10% 구간별로 분류
    bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    labels = [
        "0-10%",
        "10-20%",
        "20-30%",
        "30-40%",
        "40-50%",
        "50-60%",
        "60-70%",
        "70-80%",
        "80-90%",
        "90-100%",
    ]

    # 각 구간에 속하는 열 개수 계산
    histogram = (
        pd.cut(nan_percentages, bins=bins, labels=labels).value_counts().sort_index()
    )

    print("\nNaN 비율별 열 분포:")
    for interval, count in histogram.items():
        print(f"  {interval}: {count}개 열")

    # 상세 정보 출력 (선택적)
    print("\n열별 NaN 비율 요약:")
    print(f"  평균 NaN 비율: {nan_percentages.mean():.2f}%")
    print(f"  최소 NaN 비율: {nan_percentages.min():.2f}%")
    print(f"  최대 NaN 비율: {nan_percentages.max():.2f}%")
