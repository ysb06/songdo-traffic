# conftest.py
import os
import pytest
import pandas as pd
from typing import List, Set, Tuple, Optional
from metr.components.metr_imc import TrafficData
from metr.components.metr_imc.interpolation import (
    LinearInterpolator,
    SplineLinearInterpolator,
    TimeMeanFillInterpolator,
    ShiftFillInterpolator,
    MonthlyMeanFillInterpolator,
)

# 실제 경로/상수는 환경에 맞춰 수정하세요.
TRAFFIC_RAW_PATH = "../datasets/metr-imc/metr-imc.h5"
OUTLIER_OUTPUT_DIR = "./output/outlier_processed"
INTERPOLATION_OUTPUT_DIR = "./output/sync_processed"

# 이상치 처리된 파일 후보 리스트
PROCESSED_FILES: Set[str] = {
    "none_simple.h5",
    "in_sensor_zscore.h5",
    "hourly_in_sensor_zscore.h5",
    "mad_outlier.h5",
    "trimmed_mean.h5",
    "winsorized.h5",
}

INTERPOLATION_CLASSES: List = [
    LinearInterpolator,
    SplineLinearInterpolator,
    TimeMeanFillInterpolator,
    ShiftFillInterpolator,
    MonthlyMeanFillInterpolator,
]

@pytest.fixture(scope="session")
def max_sensors() -> Optional[int]:
    return 1


@pytest.fixture(scope="session")
def original_df() -> pd.DataFrame:
    """
    원본(이상치 처리 전) HDF 파일을 로딩하여 DataFrame을 반환하는 fixture.
    """
    if not os.path.exists(TRAFFIC_RAW_PATH):
        pytest.skip(f"원본 파일이 존재하지 않습니다: {TRAFFIC_RAW_PATH}")
    traffic_data_original = TrafficData.import_from_hdf(TRAFFIC_RAW_PATH, dtype=float)
    return traffic_data_original.data  # pd.DataFrame

@pytest.fixture(scope="session", params=PROCESSED_FILES)
def outlier_processed_df_info(request) -> Tuple[str, pd.DataFrame]:
    """
    여러 종류의 이상치 처리된 파일 후보(PROCESSED_FILES)를
    session 범위로 파라미터화하여 순회하는 fixture.
    - 반환값: (선택된 파일명, 처리 후 DataFrame)
    """
    filename = request.param
    processed_path = os.path.join(OUTLIER_OUTPUT_DIR, filename)
    if not os.path.exists(processed_path):
        pytest.skip(f"처리된 파일이 존재하지 않습니다: {processed_path}")

    traffic_data_processed = TrafficData.import_from_hdf(processed_path, dtype=float)
    return filename, traffic_data_processed.data

def interp_filenames():
    filenames = []
    for proc_filename in PROCESSED_FILES:
        for interp_class in INTERPOLATION_CLASSES:
            suffix = interp_class.__name__.lower().removesuffix("interpolator")
            filename = f"{proc_filename.split('.')[0]}-{suffix}.h5"
            filenames.append(filename)
    
    return filenames

@pytest.fixture(scope="session", params=interp_filenames())
def interpolated_df_info(request) -> Tuple[str, pd.DataFrame]:
    """
    여러 종류의 보간된 파일 후보를 session 범위로 파라미터화하여 순회하는 fixture.
    - 반환값: (선택된 파일명, 보간 후 DataFrame)
    """
    filename = request.param
    interpolated_path = os.path.join(INTERPOLATION_OUTPUT_DIR, filename)
    if not os.path.exists(interpolated_path):
        pytest.skip(f"보간된 파일이 존재하지 않습니다: {interpolated_path}")

    traffic_data_interpolated = TrafficData.import_from_hdf(interpolated_path, dtype=float)
    return filename, traffic_data_interpolated.data