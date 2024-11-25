import logging
from typing import Dict

import numpy as np
import pandas as pd
import scipy.stats as stats

logger = logging.getLogger(__name__)


class OutlierProcessor:
    def _process(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        original_count = df.count()
        df_clean = self._process(df)
        clean_count = df_clean.count()
        logger.info(f"Outliers removed: {(original_count - clean_count).sum()}")

        return df_clean


class RemovingWeirdZeroOutlierProcessor(OutlierProcessor):
    def _process(self, df: pd.DataFrame) -> pd.DataFrame:
        def extend_nans_around_zeros(series: pd.Series) -> pd.Series:
            series = series.copy()
            nan_indices = series[series.isna()].index

            for idx in nan_indices:
                idx_pos = series.index.get_loc(idx)

                i = idx_pos - 1
                while i >= 0 and series.iat[i] == 0:
                    series.iat[i] = np.nan
                    i -= 1

                i = idx_pos + 1
                while i < len(series) and series.iat[i] == 0:
                    series.iat[i] = np.nan
                    i += 1

            return series

        return df.apply(extend_nans_around_zeros)


class SimpleAbsoluteOutlierProcessor(OutlierProcessor):
    def __init__(self, threshold: float = 8000) -> None:
        self.threshold = threshold

    def _process(self, df: pd.DataFrame) -> pd.DataFrame:
        df_clean = df.mask(np.abs(df) > self.threshold)
        return df_clean


class TrafficCapacityAbsoluteOutlierProcessor:
    def __init__(
        self,
        road_speed_limits: Dict[str, int],
        lane_counts: Dict[str, int],
        adjustment_rate: float = 1.0,
    ) -> None:
        self.road_speed_limits = road_speed_limits
        self.lane_counts = lane_counts
        self.adjustment_rate = adjustment_rate

    def _get_road_capacity(self, road_name: str) -> float:
        speed_limit = self.road_speed_limits[road_name]
        lane_count = self.lane_counts[road_name]
        # 이상적인 허용 용량 계산 (2200 - 10 * (100 - 속도제한)) * 차선 수 * 비율
        return (2200 - 10 * (100 - speed_limit)) * lane_count * self.adjustment_rate

    def _process_road_data(self, series: pd.Series) -> pd.Series:
        road_name = series.name
        road_capacity = self._get_road_capacity(road_name)
        return series.mask(series > road_capacity)

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.apply(self._process_road_data)


class SimpleZscoreOutlierProcessor:
    def __init__(self, threshold: float = 5) -> None:
        self.threshold = threshold

    def _detect_outliers_zscore(self, df: pd.DataFrame) -> pd.DataFrame:
        z_scores = stats.zscore(df, nan_policy="omit")
        outliers = np.abs(z_scores) > self.threshold
        return outliers

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        outliers = self._detect_outliers_zscore(df)
        df_clean = df.mask(outliers)
        return df_clean


class HourlyInSensorZscoreOutlierProcessor(OutlierProcessor):
    def __init__(self, threshold: float = 5) -> None:
        self.threshold = threshold

    def _process(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.apply(self._apply_threshold_to_series)

    def _apply_threshold_to_series(self, series: pd.Series) -> pd.Series:
        z_scores = stats.zscore(series, nan_policy="omit")
        series[np.abs(z_scores) > self.threshold] = np.nan

        return series


class HourlyZscoreOutlierProcessor(OutlierProcessor):
    def __init__(self, threshold: float = 5) -> None:
        self.threshold = threshold

    def _detect_outliers_zscore_df(self, df: pd.DataFrame, threshold=5) -> pd.DataFrame:
        """시간대 별로 데이터를 그룹화하여 z-score 계산"""
        outliers = pd.DataFrame(index=df.index, columns=df.columns, dtype=bool)

        info = []
        for hour in range(24):
            hourly_data: pd.DataFrame = df[df.index.hour == hour]

            mean = hourly_data.stack().mean()  # 시간대별 모든 값의 평균
            std = hourly_data.stack().std()  # 시간대별 모든 값의 표준편차
            z_scores = (hourly_data - mean) / std

            outliers.loc[hourly_data.index] = np.abs(z_scores) > threshold
            info.append(
                {
                    "hour": hour,
                    "mean": mean,
                    "std": std,
                    "threshold": threshold,
                    "outlier_threshold": mean + threshold * std,
                }
            )

        outliers = outliers.convert_dtypes()
        dtypes = outliers.dtypes.unique()
        if len(dtypes) > 1:
            print("Warning: Different data types detected in the outlier dataframe.")
        return outliers, info

    def _process(self, df: pd.DataFrame) -> pd.DataFrame:
        outliers, _ = self._detect_outliers_zscore_df(df, self.threshold)
        df_clean = df.mask(outliers)

        return df_clean

    # Todo: Complete Components
