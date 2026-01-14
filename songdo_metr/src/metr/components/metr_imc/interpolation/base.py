import pandas as pd

import logging

logger = logging.getLogger(__name__)


class Interpolator:
    def __init__(self) -> None:
        self.name = self.__class__.__name__

    def _interpolate(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def interpolate(self, df: pd.DataFrame) -> pd.DataFrame:
        return self._interpolate(df.copy())


class LinearInterpolator(Interpolator):
    def _interpolate(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.interpolate(method="linear", axis=0)


class SplineLinearInterpolator(Interpolator):
    def _interpolate(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.interpolate(method="slinear", axis=0)


class TimeMeanFillInterpolator(Interpolator):
    def _fill_na_with_same_time(self, s: pd.Series) -> pd.Series:
        s_filled = s.copy()
        for hour in range(24):
            hourly_data = s[s.index.hour == hour]
            mean_value = hourly_data.mean(skipna=True)
            s_filled.loc[s_filled.index.hour == hour] = s_filled.loc[
                s_filled.index.hour == hour
            ].fillna(mean_value)

        return s_filled
        # 기존의 값이 이상한 부분이 있었다. 현재가 정확할 수 있다

    def _interpolate(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.apply(self._fill_na_with_same_time, axis=0)
        result = result.round()
        return result


class ShiftFillInterpolator(Interpolator):
    def __init__(self, periods: int = 168) -> None:
        self.periods = periods

    def _fill_na_with_shifted(self, s: pd.Series) -> pd.Series:
        """결측치를 과거 데이터로 채우는 최적화된 함수"""
        # 결측치가 없으면 빠르게 반환
        if not s.isna().any():
            return s

        s_filled = s.copy()
        na_indices = s[s.isna()].index
        values_dict = s.to_dict()
        min_idx = s.index.min()

        # 각 결측치에 대해 처리
        for na_idx in na_indices:
            shifted_time_idx = na_idx - pd.Timedelta(hours=self.periods)
            while shifted_time_idx >= min_idx:
                # 해당 시점에 값이 있고 NaN이 아니면 사용
                if shifted_time_idx in values_dict and not pd.isna(
                    values_dict[shifted_time_idx]
                ):
                    s_filled.loc[na_idx] = values_dict[shifted_time_idx]
                    break
                # 아닌 경우 다음 기간으로 이동
                shifted_time_idx -= pd.Timedelta(hours=self.periods)

        return s_filled

    def _interpolate(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.apply(self._fill_na_with_shifted, axis=0)
        return result


class MonthlyMeanFillInterpolator(Interpolator):
    def _interpolate(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()

        for col in result.columns:
            result_index: pd.DatetimeIndex = result.index
            groups = result[col].groupby([result_index.year, result_index.month])

            # 각 그룹별 평균 계산
            group_means: pd.Series = groups.mean()

            for key, mean_value in group_means.items():
                yr, mon = key
                mask = (
                    (result_index.year == yr)
                    & (result_index.month == mon)
                    & (result[col].isna())
                )

                result.loc[mask, col] = mean_value

        return result
