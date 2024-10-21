import os
from typing import Dict

from ..outliers.conftest import Configs
from metr.components import (
    TrafficData,
    Metadata,
    IdList,
    SensorLocations,
    AdjacencyMatrix,
    DistancesImc,
)
import geopandas as gpd
import matplotlib.pyplot as plt
from folium.folium import Map
from metr.components.metr_imc.outlier import (
    SimpleAbsoluteOutlierProcessor,
    TrafficCapacityAbsoluteOutlierProcessor,
    HourlyZscoreOutlierProcessor,
    SimpleZscoreOutlierProcessor,
    HourlyInSensorZscoreOutlierProcessor,
)
import pytest


@pytest.mark.run(order=5)
def test_simple_absolute_outlier(
    selected_training_traffic_data: TrafficData,
    outlier_output_path: Dict[str, str],
):
    threshold = 8000
    processor = SimpleAbsoluteOutlierProcessor(threshold)
    selected_training_traffic_data.remove_outliers(processor)
    selected_training_traffic_data.to_hdf(outlier_output_path["simple_absolute"])


@pytest.mark.run(order=5)
def test_traffic_capacity_absolute_outlier(
    selected_training_traffic_data: TrafficData,
    road_metadata: Metadata,
    outlier_output_path: Dict[str, str],
):
    speed_limit_map = road_metadata.data.set_index("LINK_ID")["MAX_SPD"].to_dict()
    lane_map = road_metadata.data.set_index("LINK_ID")["LANES"].to_dict()
    processor = TrafficCapacityAbsoluteOutlierProcessor(speed_limit_map, lane_map)

    selected_training_traffic_data.remove_outliers(processor)
    selected_training_traffic_data.to_hdf(
        outlier_output_path["traffic_capacity_absolute"]
    )


@pytest.mark.run(order=5)
def test_simple_zscore_outlier(
    selected_training_traffic_data: TrafficData,
    outlier_output_path: Dict[str, str],
):
    """
    Z-score 기반의 outlier를 처리하는 테스트.
    시간대별 구분 없이 전체 데이터를 대상으로 z-score를 구해 처리합니다.
    """
    threshold = 5.0  # 임계값
    processor = SimpleZscoreOutlierProcessor(threshold)
    selected_training_traffic_data.remove_outliers(processor)
    selected_training_traffic_data.to_hdf(outlier_output_path["simple_zscore"])


@pytest.mark.run(order=5)
def test_hourly_zscore_outlier(
    selected_training_traffic_data: TrafficData,
    outlier_output_path: Dict[str, str],
):
    """
    시간대 별 z-score 기반의 outlier 처리 테스트.
    """
    threshold = 5.0  # 임계값
    processor = HourlyZscoreOutlierProcessor(threshold)
    selected_training_traffic_data.remove_outliers(processor)
    selected_training_traffic_data.to_hdf(outlier_output_path["hourly_zscore"])


@pytest.mark.run(order=5)
def test_hourly_in_sensor_zscore_outlier(
    selected_training_traffic_data: TrafficData,
    outlier_output_path: Dict[str, str],
):
    """
    각 센서별로 시간대에 따라 z-score를 계산하여 outlier 처리 테스트.
    """
    threshold = 5.0  # 임계값
    processor = HourlyInSensorZscoreOutlierProcessor(threshold)
    selected_training_traffic_data.remove_outliers(processor)
    selected_training_traffic_data.to_hdf(
        outlier_output_path["hourly_in_sensor_zscore"]
    )
