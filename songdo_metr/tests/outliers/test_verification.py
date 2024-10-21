import os
from typing import Dict
import pandas as pd
import numpy as np
from scipy import stats

from metr.components.metadata import Metadata
from metr.components.metr_imc.traffic_data import TrafficData


# 원본과 처리된 데이터를 불러오는 함수
def load_traffic_data(file_path: str) -> pd.DataFrame:
    traffic_data = pd.read_hdf(file_path)
    return traffic_data


# Simple Absolute Outlier Verification
def test_verify_simple_absolute_outlier(
    selected_training_traffic_data: TrafficData,
    outlier_output_path: Dict[str, str],
    threshold: float = 8000,
):
    original_df = selected_training_traffic_data.data
    processed_df = pd.read_hdf(outlier_output_path["simple_absolute"])

    # Outlier는 threshold를 초과한 값이어야 함
    # 처리된 데이터에서 outliers가 NaN으로 처리되었는지 확인
    outliers = original_df[(original_df > threshold)]
    removed_outliers = original_df.mask(processed_df.notna())

    assert (
        (outliers == removed_outliers).all().all()
    ), "Simple Absolute Outlier 검증 실패"


# Traffic Capacity Outlier Verification
def test_verify_traffic_capacity_outlier(
    selected_training_traffic_data: TrafficData,
    road_metadata: Metadata,
    outlier_output_path: Dict[str, str],
):
    original_df = selected_training_traffic_data.data
    processed_df = pd.read_hdf(outlier_output_path["traffic_capacity_absolute"])

    speed_limit_map = road_metadata.data.set_index("LINK_ID")["MAX_SPD"].to_dict()
    lane_map = road_metadata.data.set_index("LINK_ID")["LANES"].to_dict()

    def calculate_capacity(road_name: str) -> float:
        speed_limit = speed_limit_map[road_name]
        lane_count = lane_map[road_name]
        return (2200 - 10 * (100 - speed_limit)) * lane_count

    outliers = pd.DataFrame()
    for column in original_df.columns:
        road_capacity = calculate_capacity(column)
        road_outliers = original_df[original_df[column] > road_capacity][column]
        outliers[column] = road_outliers

    removed_outliers = original_df.mask(processed_df.notna())

    assert (
        (outliers == removed_outliers).all().all()
    ), "Traffic Capacity Outlier 검증 실패"


# Z-Score Outlier Verification (Simple)
def test_verify_simple_zscore_outlier(
    selected_training_traffic_data: TrafficData,
    outlier_output_path: Dict[str, str],
    threshold: float = 5.0,
):
    original_df = selected_training_traffic_data.data
    processed_df = pd.read_hdf(outlier_output_path["simple_zscore"])

    # z-score 계산
    z_scores = stats.zscore(original_df, nan_policy="omit")

    outliers = pd.DataFrame(
        np.abs(z_scores) > threshold,
        index=original_df.index,
        columns=original_df.columns,
    )

    removed_outliers = original_df.mask(processed_df.notna())

    assert (
        (outliers == removed_outliers.notna()).all().all()
    ), "Simple Z-Score Outlier 검증 실패"


# Hourly Z-Score Outlier Verification
def test_verify_hourly_zscore_outlier(
    selected_training_traffic_data: TrafficData,
    outlier_output_path: Dict[str, str],
    threshold: float = 5.0,
):
    original_df = selected_training_traffic_data.data
    processed_df = pd.read_hdf(outlier_output_path["hourly_zscore"])

    outliers = pd.DataFrame(index=original_df.index, columns=original_df.columns)

    # 시간대별 z-score 계산 및 outlier 확인
    for hour in range(24):
        hourly_data = original_df[original_df.index.hour == hour]
        z_scores = (hourly_data - hourly_data.mean()) / hourly_data.std()
        hourly_outliers = np.abs(z_scores) > threshold
        outliers.loc[hourly_outliers.index] = hourly_outliers

    removed_outliers = original_df.mask(processed_df.notna())

    assert (
        (outliers == removed_outliers.notna()).all().all()
    ), "Hourly Z-Score Outlier 검증 실패"


# Hourly In-Sensor Z-Score Verification
def test_verify_hourly_in_sensor_zscore_outlier(
    selected_training_traffic_data: TrafficData,
    outlier_output_path: Dict[str, str],
    threshold: float = 5.0,
):
    threshold: float = 5.0
    original_df = selected_training_traffic_data.data
    processed_df = pd.read_hdf(outlier_output_path["hourly_in_sensor_zscore"])

    outliers = pd.DataFrame(index=original_df.index, columns=original_df.columns)

    # 각 센서별로 시간대별 z-score 계산 및 outlier 확인
    for column in original_df.columns:
        for hour in range(24):
            hourly_data = original_df[original_df.index.hour == hour][column]
            z_scores = stats.zscore(hourly_data, nan_policy="omit")
            sensor_outliers = np.abs(z_scores) > threshold
            outliers.loc[hourly_data.index, column] = sensor_outliers

    removed_outliers = original_df.mask(processed_df.notna())

    assert (
        (outliers == removed_outliers.notna()).all().all()
    ), "Hourly In-Sensor Z-Score Outlier 검증 실패"
