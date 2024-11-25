import os

import numpy as np
import pandas as pd
from metr.components import TrafficData
from metr.processors import (
    TRAFFIC_FILENAME,
    TRAFFIC_TRAINING_FILENAME,
    TRAFFIC_TEST_FILENAME,
    METADATA_FILENAME,
    SENSOR_IDS_FILENAME,
    SENSOR_LOCATIONS_FILENAME,
    DISTANCES_FILENAME,
    ADJ_MX_FILENAME,
)

OLD_VERSION_ROOT = "../datasets/metr-imc_legacy"
OLD_METR_IMC_PATH = os.path.join(OLD_VERSION_ROOT, TRAFFIC_FILENAME)
OLD_METR_IMC_TRAINING_FILENAME = os.path.join(OLD_VERSION_ROOT, TRAFFIC_TRAINING_FILENAME)
OLD_METR_IMC_TEST_FILENAME = os.path.join(OLD_VERSION_ROOT, TRAFFIC_TEST_FILENAME)
OLD_METADATA_FILENAME = os.path.join(OLD_VERSION_ROOT, METADATA_FILENAME)
OLD_SENSOR_IDS_FILENAME = os.path.join(OLD_VERSION_ROOT, SENSOR_IDS_FILENAME)
OLD_SENSOR_LOCATIONS_FILENAME = os.path.join(OLD_VERSION_ROOT, SENSOR_LOCATIONS_FILENAME)
OLD_DISTANCES_FILENAME = os.path.join(OLD_VERSION_ROOT, DISTANCES_FILENAME)
OLD_ADJ_MX_FILENAME = os.path.join(OLD_VERSION_ROOT, ADJ_MX_FILENAME)

NEW_VERSION_ROOT = "../datasets/metr-imc"
NEW_METR_IMC_PATH = os.path.join(NEW_VERSION_ROOT, TRAFFIC_FILENAME)
NEW_METR_IMC_TRAINING_FILENAME = os.path.join(NEW_VERSION_ROOT, TRAFFIC_TRAINING_FILENAME)
NEW_METR_IMC_TEST_FILENAME = os.path.join(NEW_VERSION_ROOT, TRAFFIC_TEST_FILENAME)
NEW_METADATA_FILENAME = os.path.join(NEW_VERSION_ROOT, METADATA_FILENAME)
NEW_SENSOR_IDS_FILENAME = os.path.join(NEW_VERSION_ROOT, SENSOR_IDS_FILENAME)
NEW_SENSOR_LOCATIONS_FILENAME = os.path.join(NEW_VERSION_ROOT, SENSOR_LOCATIONS_FILENAME)
NEW_DISTANCES_FILENAME = os.path.join(NEW_VERSION_ROOT, DISTANCES_FILENAME)
NEW_ADJ_MX_FILENAME = os.path.join(NEW_VERSION_ROOT, ADJ_MX_FILENAME)

def test_compare_traffic_data():
    # 데이터 로드
    old_metr_imc = TrafficData.import_from_hdf(OLD_METR_IMC_PATH, dtype=float)
    new_metr_imc = TrafficData.import_from_hdf(NEW_METR_IMC_PATH, dtype=float)

    # Index 비교
    assert old_metr_imc.data.index.equals(new_metr_imc.data.index), "Index가 동일하지 않습니다."

    # Column 비교
    assert len(old_metr_imc.data.columns.difference(new_metr_imc.data.columns)) == 0, "Column이 동일하지 않습니다."

    # 값 차이 계산
    diff = old_metr_imc.data - new_metr_imc.data
    unequal_mask = diff != 0

    # 값이 다른 데이터만 선택
    unequal_values = diff[unequal_mask]

    # 값이 다른 경우 개수와 차이 출력
    num_differences = unequal_values.count().sum()  # 차이가 있는 값의 총 개수
    max_difference = unequal_values.abs().max().max()  # 가장 큰 차이
    mean_difference = unequal_values.abs().mean().mean()  # 평균 차이

    # 테스트 조건
    assert num_differences == 0, (
        f"값이 다른 항목이 {num_differences}개 발견되었습니다.\n"
        f"가장 큰 차이: {max_difference}\n"
        f"평균 차이: {mean_difference}"
    )

    # 결과 출력 (테스트 디버깅용)
    if num_differences > 0:
        print(f"값이 다른 항목이 {num_differences}개 발견되었습니다.")
        print(f"가장 큰 차이: {max_difference}")
        print(f"평균 차이: {mean_difference}")
        print("다른 값들:")
        print(unequal_values.dropna(how="all"))  # 값이 다른 부분만 출력