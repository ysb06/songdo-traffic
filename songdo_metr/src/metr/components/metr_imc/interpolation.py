from typing import Literal, Tuple, Union

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from tqdm import tqdm
from joblib import Parallel, delayed

import logging
logger = logging.getLogger(__name__)

class Interpolator:
    def interpolate(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError


class LinearInterpolator(Interpolator):
    def interpolate(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.interpolate(method="linear", axis=0)


class SplineLinearInterpolator(Interpolator):
    def interpolate(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.interpolate(method="slinear", axis=0)


class SplineInterpolator:
    def __init__(self, order: int = 3, n_jobs=1) -> None:
        self.order = order
        self.n_jobs = n_jobs

    def interpolate(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self._interpolate_sensor)(df[col_name], col_name)
            for col_name in tqdm(df.columns)
        )
        for col_name, interpolated_col in results:
            df[col_name] = interpolated_col

        return df

    def _interpolate_sensor(self, series: pd.Series, col_name: str):
        interpolated_col = series.interpolate(method="spline", order=self.order)
        return col_name, interpolated_col


class TimeMeanFillInterpolator(Interpolator):
    "한 열의 결측치를 한 열 내 같은 시간대의 평균으로 채우는 Interpolator"

    def _fill_na_with_same_time(self, s: pd.Series) -> pd.Series:
        s_filled = s.copy()
        for hour in range(24):
            hourly_data = s[s.index.hour == hour]
            mean_value = hourly_data.mean(skipna=True)
            if pd.isna(mean_value):
                logger.warning(f"Problem in [{s.name}] in time [{str(hour)}]")
                logger.warning("Mean value will be forced to set as 0")
                mean_value = 0
            s_filled.loc[s_filled.index.hour == hour] = s_filled.loc[
                s_filled.index.hour == hour
            ].fillna(mean_value)

        return s_filled
        # 기존의 값이 이상한 부분이 있었다. 현재가 정확할 수 있다

    def interpolate(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.apply(self._fill_na_with_same_time, axis=0)
        result = result.round()
        result = result.astype(int)
        return result


# 추가로 논문을 찾아 보간법을 구현


# 아래 보간법은 폐기. 시계열 데이터에 맞는 보간법이 아님.
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
