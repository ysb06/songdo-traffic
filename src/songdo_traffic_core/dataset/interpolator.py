from typing import Literal, Tuple, Union
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import pandas as pd


class InterpolatorBase:
    def interpolate(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError


class IterativeRandomForestInterpolator(InterpolatorBase):
    def __init__(
        self,
        random_state: Union[int, Tuple[int, int]] = 42,
        max_iter: int = 5,
        imputation_order: Literal[
            "ascending", "descending", "roman", "arabic", "random"
        ] = "ascending",
        verbose: int = 5,
    ) -> None:
        if isinstance(random_state, int):
            random_state = (random_state, random_state)

        self.estimator = RandomForestRegressor(
            random_state=random_state[0], n_jobs=-1, verbose=50
        )
        self.iterative_imputer = IterativeImputer(
            estimator=self.estimator,
            verbose=verbose,
            imputation_order=imputation_order,
            random_state=random_state[1],
        )

    def interpolate(self, df: pd.DataFrame) -> pd.DataFrame:
        fitted_data = self.iterative_imputer.fit_transform(df)
        return pd.DataFrame(fitted_data, columns=df.columns, index=df.index)


class LinearInterpolator(InterpolatorBase):
    def interpolate(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.interpolate(method="linear", axis=0)


class TotalMeanFillInterpolator(InterpolatorBase):
    def interpolate(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.fillna(df.mean())


class ColumnMeanFillInterpolator(InterpolatorBase):
    def __fill_column_mean(self, column: pd.Series):
        return column.fillna(column.mean())

    def interpolate(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.apply(self.__fill_column_mean, axis=0)
