from typing import Literal, Optional, Tuple, Union
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import pandas as pd
from tqdm import tqdm


class Interpolator:
    def interpolate(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError


class IterativeRandomForestInterpolator(Interpolator):
    def __init__(
        self,
        random_state: Union[int, Tuple[int, int]] = 42,
        max_iter: int = 10,
        imputation_order: Literal[
            "ascending", "descending", "roman", "arabic", "random"
        ] = "ascending",
        verbose: int = 5,
        n_jobs: int = 8,
    ) -> None:
        if isinstance(random_state, int):
            random_state = (random_state, random_state)

        self.estimator = RandomForestRegressor(
            random_state=random_state[0], n_jobs=n_jobs, verbose=verbose
        )
        self.iterative_imputer = IterativeImputer(
            estimator=self.estimator,
            verbose=verbose,
            imputation_order=imputation_order,
            random_state=random_state[1],
            max_iter=max_iter,
        )

    def interpolate(self, df: pd.DataFrame) -> pd.DataFrame:
        fitted_data = self.iterative_imputer.fit_transform(df)
        return pd.DataFrame(fitted_data, columns=df.columns, index=df.index)


class LinearInterpolator(Interpolator):
    def interpolate(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.interpolate(method="linear", axis=0, limit_direction="both")


class SplineInterpolator(Interpolator):
    def interpolate(self, df: pd.DataFrame, need_round: bool = True) -> pd.DataFrame:
        df = df.interpolate(method="spline", order=3, axis=0)
        if need_round:
            return df.round().astype(int)
        else:
            return df


class SmartSplineInterpolator(SplineInterpolator):
    def interpolate(self, df: pd.DataFrame) -> pd.DataFrame:
        tqdm.pandas()
        df = super().interpolate(df)
        return df.apply(self.__fill_na_with_same_time, axis=1)

    def __fill_na_with_same_time(self, s: pd.Series) -> pd.Series:
        s_filled = s.copy()
        for hour in range(24):
            hourly_data = s[s.index.hour == hour]
            mean_value = hourly_data.mean(skipna=True)
            s_filled.loc[s_filled.index.hour == hour] = s_filled.loc[
                s_filled.index.hour == hour
            ].fillna(mean_value)
        return s_filled


class TotalMeanFillInterpolator(Interpolator):
    def interpolate(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.fillna(df.mean())


class ColumnMeanFillInterpolator(Interpolator):
    def __fill_column_mean(self, column: pd.Series):
        return column.fillna(column.mean())

    def interpolate(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.apply(self.__fill_column_mean, axis=1)


# 이것 외 다른 Interpolator는 수정해도 무방


class TimeMeanFillInterpolator(Interpolator):
    "한 열의 결측치를 한 열 내 같은 시간대의 평균으로 채우는 Interpolator"

    def __fill_na_with_same_time(self, s: pd.Series) -> pd.Series:
        s_filled = s.copy()
        for hour in range(24):
            hourly_data = s[s.index.hour == hour]
            mean_value = hourly_data.mean(skipna=True)
            s_filled.loc[s_filled.index.hour == hour] = s_filled.loc[
                s_filled.index.hour == hour
            ].fillna(mean_value)
        return s_filled

    def interpolate(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.apply(self.__fill_na_with_same_time, axis=0).round().astype(int)


if __name__ == "__main__":
    pass
