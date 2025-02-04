# test_interpolation_visual.py

import pytest
import os
import matplotlib.pyplot as plt
import pandas as pd

from metr.components import TrafficData


def test_interpolation_visual(
    outlier_processed_df_info,  # (파일명, 이상치 처리 후 DataFrame)
    interpolated_df_info,  # (파일명, 보간 완료 후 DataFrame)
    max_sensors,
):
    """
    이상치 처리 후(원본) vs 보간 후 데이터를 한 그래프에 시각화.
    - 원래 결측치: 노란 점
    - 보간 후에도 남아 있는 결측치: 빨간 점
    - 원본 데이터 (검정선), 보간된 데이터 (파란선)
    """
    # unpack
    _, df_processed = outlier_processed_df_info
    int_filename, df_interpolated = interpolated_df_info

    # 최대 시각화할 센서 개수
    sensors = df_processed.columns
    if max_sensors is not None and max_sensors < len(sensors):
        sensors = sensors[:max_sensors]

    for sensor in sensors:
        # 혹시 보간 결과에 해당 센서가 없으면 스킵
        if sensor not in df_interpolated.columns:
            continue

        # 원본(이상치 처리 후) 시리즈와 보간 후 시리즈
        s_orig = df_processed[sensor]
        s_interp = df_interpolated[sensor]

        # 시각화를 위한 DataFrame 구성
        df_plot = pd.DataFrame({"orig": s_orig, "interp": s_interp})

        # [1] 원래 결측치(노란색)
        orig_missing_mask = df_plot["orig"].isna()

        # [2] 보간 후에도 남아 있는 결측치(빨간색)
        interp_missing_mask = df_plot["interp"].isna()

        fig, ax = plt.subplots(figsize=(14, 4))

        # (A) 선 그래프: 원본(검정), 보간(파랑)
        ax.plot(
            df_plot.index,
            df_plot["orig"],
            color="black",
            linewidth=0.8,
            label="Original",
        )
        ax.plot(
            df_plot.index,
            df_plot["interp"],
            color="blue",
            linewidth=0.8,
            label="Interpolated",
        )

        # (B) 원래 결측치(노란 점)
        ax.scatter(
            df_plot.index[orig_missing_mask],
            [0] * orig_missing_mask.sum(),  # y=0 위치에 표시
            color="yellow",
            s=10,
            label="Original Missing",
        )

        # (C) 보간 후에도 남아 있는 결측치(빨간 점)
        ax.scatter(
            df_plot.index[interp_missing_mask],
            [0] * interp_missing_mask.sum(),  # y=0 위치
            color="red",
            s=10,
            label="Still Missing (Interpolated)",
        )

        ax.set_title(f"{int_filename} - Sensor={sensor}")
        ax.legend()
        plt.tight_layout()
        plt.show()
        plt.close(fig)
