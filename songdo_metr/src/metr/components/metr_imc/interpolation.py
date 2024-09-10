import pandas as pd


class Interpolator:
    def interpolate(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError


class TimeMeanFillInterpolator(Interpolator):
    "한 열의 결측치를 한 열 내 같은 시간대의 평균으로 채우는 Interpolator"

    def __fill_na_with_same_time(self, s: pd.Series) -> pd.Series:
        s_filled = s.copy()
        for hour in range(24):
            hourly_data = s[s.index.hour == hour]
            mean_value = hourly_data.mean(skipna=True)
            if pd.isna(mean_value):
                print("Problem in [", s.name, "] in time[", hour, "]")
            s_filled.loc[s_filled.index.hour == hour] = s_filled.loc[
                s_filled.index.hour == hour
            ].fillna(mean_value)

        return s_filled

    def interpolate(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.apply(self.__fill_na_with_same_time, axis=0)
        result = result.round()
        result = result.astype(int)
        return result
