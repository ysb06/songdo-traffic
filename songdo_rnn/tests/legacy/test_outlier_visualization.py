# test_outlier_visual.py
from typing import Tuple
import pandas as pd
import matplotlib.pyplot as plt

def test_outlier_visualization(
    original_df: pd.DataFrame,
    outlier_processed_df_info: Tuple[str, pd.DataFrame],
    max_sensors: int
):
    """
    Pytest에서 원본 데이터와 이상치 처리 후 데이터를 비교 시각화하는 예시 테스트.
    - original_df: conftest.py의 원본 데이터프레임 fixture
    - processed_df: (파일명, 처리 후 데이터프레임) 튜플
    - max_sensors: 시각화할 센서 최대 개수 (None이면 전체)
    """
    # unpack
    filename, processed_df = outlier_processed_df_info

    print(f"\n=== 현재 테스트 중인 파일: {filename} ===")    
    sensors = original_df.columns
    
    if max_sensors is not None and max_sensors < len(sensors):
        sensors = sensors[:max_sensors]
        print(f"센서가 많아 {max_sensors}개까지만 시각화합니다.")

    df_orig_segmented = original_df.loc[processed_df.index.intersection(original_df.index)]
    for sensor in sensors:
        s_orig = df_orig_segmented[sensor]
        s_proc = processed_df[sensor]

        # 시각화를 위한 DataFrame
        df_plot = pd.DataFrame({
            "orig": s_orig,
            "proc": s_proc,
        })

        # 결측치 여부 구분
        df_plot["orig_missing"] = df_plot["orig"].isna()
        df_plot["new_missing"] = (~df_plot["orig_missing"]) & (df_plot["proc"].isna())

        # 그래프 생성
        fig, ax = plt.subplots(figsize=(14, 4))

        ax.plot(
            df_plot.index, df_plot["orig"],
            color="black", linewidth=0.8, label="Original"
        )
        ax.plot(
            df_plot.index, df_plot["proc"],
            color="blue", linewidth=0.8, label="Processed"
        )

        # 원래 결측치 (노란색)
        orig_missing_mask = df_plot["orig_missing"]
        ax.scatter(
            df_plot.index[orig_missing_mask],
            [0]*orig_missing_mask.sum(),
            color="yellow", s=10, label="Original Missing"
        )

        # 새로 결측치 (빨간색)
        new_missing_mask = df_plot["new_missing"]
        ax.scatter(
            df_plot.index[new_missing_mask],
            [0]*new_missing_mask.sum(),
            color="red", s=10, label="New Missing (Outlier)"
        )

        ax.legend(loc="upper right")
        ax.set_title(f"[{filename}] Sensor: {sensor} | Original vs Processed")
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")

        plt.tight_layout()
        plt.show()
        plt.close(fig)

    print(f"파일 '{filename}'의 센서 시각화를 모두 완료했습니다.")