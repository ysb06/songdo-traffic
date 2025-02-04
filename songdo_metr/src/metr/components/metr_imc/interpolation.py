import pandas as pd

import logging

logger = logging.getLogger(__name__)


class Interpolator:
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
    def __init__(self, periods: int = 1) -> None:
        self.periods = periods

    def _interpolate(self, df: pd.DataFrame) -> pd.DataFrame:
        df_index: pd.DatetimeIndex = df.index
        df_shifted = df.shift(periods=self.periods, freq=df_index.freq)
        filled_df = df.where(~df.isna(), df_shifted)
        return filled_df


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