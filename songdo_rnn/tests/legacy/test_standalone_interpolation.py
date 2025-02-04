# test_standalone_interpolation.py

import os
from typing import List, Set
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from metr.components.metr_imc import TrafficData
from metr.components.metr_imc.interpolation import (
    Interpolator,
    LinearInterpolator,
    MonthlyMeanFillInterpolator,
    ShiftFillInterpolator,
    SplineLinearInterpolator,
    TimeMeanFillInterpolator,
)
from glob import glob
from pprint import pprint


@pytest.fixture()
def sample_df() -> pd.DataFrame:
    raw = TrafficData.import_from_hdf("./output/outlier_processed/none_simple.h5")
    return raw.data


@pytest.fixture()
def interpolator() -> Interpolator:
    # return MonthlyMeanFillInterpolator()
    return LinearInterpolator()


@pytest.fixture()
def processed_dir():
    return "./output/all_processed"


@pytest.fixture()
def sensor_count():
    return -1


def test_sync(processed_dir: str):
    traffic_files = glob(os.path.join(processed_dir, "*.h5"))

    # non_nan_sensors: List[Set[str]] = []
    results = []
    for file_path in traffic_files:
        traffic_data = TrafficData.import_from_hdf(file_path)
        columns = set(traffic_data.data.columns[traffic_data.data.notna().all()])
        result = f"{len(columns):>5d}: {file_path}"
        outlier_proc = os.path.basename(file_path).split("-")[0]
        interp_proc = os.path.basename(file_path).split("-")[1]
        results.append((result, outlier_proc, interp_proc))
        # non_nan_sensors.append(columns)
    
    print("List")
    pprint([result for result, _, _ in sorted(results, key=lambda x: x[2])])


def test_interpolation_standalone(
    sample_df: pd.DataFrame, interpolator: Interpolator, sensor_count: int
):
    df_interpolated = interpolator.interpolate(sample_df.copy())
    notna_cols = df_interpolated.columns[df_interpolated.notna().all()]
    print("Not NaN Columns:", len(notna_cols))
    isna_cols = df_interpolated.columns[df_interpolated.isna().all()]
    print("Is NaN Columns:", len(isna_cols))

    # 센서(컬럼)별로 그래프 그리기
    count = 0
    for sensor in sample_df.columns:
        s_orig = sample_df[sensor]
        s_interp = df_interpolated[sensor]

        interp_result = s_interp.loc["2024-01-01 00:00:00":]
        if interp_result.isna().sum() == 0:
            continue
        
        count += 1
        if sensor_count > 0 and count > sensor_count:
            break
        # 시각화를 위해 DF 결합
        df_plot = pd.DataFrame({"orig": s_orig, "interp": s_interp})

        # 원본 데이터에서 NaN 위치 파악 (빨간 점)
        missing_mask = df_plot["orig"].isna()
        interp_missing_mask = df_plot["interp"].isna()

        # 그래프 생성
        fig, ax = plt.subplots(figsize=(12, 3))

        # 선 그래프: 원본(검정), 보간 후(파랑)
        ax.plot(
            df_plot.index,
            df_plot["orig"],
            color="black",
            linewidth=1.2,
            label="Original",
        )
        ax.plot(
            df_plot.index,
            df_plot["interp"],
            color="blue",
            linewidth=1.2,
            label="Interpolated",
        )

        # NaN 위치에 빨간 점 표시
        ax.scatter(
            df_plot.index[missing_mask],
            [0] * missing_mask.sum(),  # y=0 위치에 표시
            color="green",
            s=30,
            label="Original Missing",
        )

        ax.scatter(
            df_plot.index[interp_missing_mask],
            [0] * interp_missing_mask.sum(),  # y=0 위치
            color="red",
            s=10,
            label="Still Missing (Interpolated)",
        )

        # 그래프 설정
        ax.set_title(f"{interpolator.__class__.__name__} - Sensor: {sensor}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.legend(loc="best")

        plt.tight_layout()
        plt.show()
        plt.close(fig)

    # 추가 검증 등
    # 예: 보간 결과에 NaN이 얼마나 남았는지 체크
    remaining_nans = df_interpolated.isna().sum().sum()
    print(f"보간 후 남아있는 NaN 수: {remaining_nans}")
    # (필요에 따라 assert 로직 추가 가능)
