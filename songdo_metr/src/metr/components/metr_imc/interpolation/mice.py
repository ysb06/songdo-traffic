import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.exceptions import ConvergenceWarning
from tqdm import tqdm
from joblib import Parallel, delayed
import warnings

from .base import Interpolator


class MICEInterpolator(Interpolator):
    """
    MICE (Multivariate Imputation by Chained Equations) Interpolator

    각 센서를 독립적으로 처리하여 결측치를 보간합니다.
    센서가 많은 경우(~2000개) 연산량과 노이즈 유입을 방지하기 위한 베이스라인 구현입니다.

    Features:
    - Temporal patterns (hour-of-day, day-of-week) with cyclic encoding
    - Time-lag features (lag 1, 2, 3)
    - Spatial proxy: Network-wide average traffic at each timestamp
    """

    def __init__(
        self,
        n_estimators: int = 10,
        max_iter: int = 16,
        random_state: int = 42,
        verbose: int = 0,
        fallback_method: str = "linear",
        n_jobs: int = -3,
        suppress_warnings: bool = True,
        track_warnings: bool = True,
    ) -> None:
        """
        Args:
            n_estimators: Number of trees in ExtraTreesRegressor
            max_iter: Maximum MICE iterations
            random_state: Random seed for reproducibility
            verbose: Logging level (0=silent, 1=progress, 2=detailed)
            fallback_method: Fallback interpolation for remaining NaNs
                           ('linear', 'ffill', 'bfill', 'median')
            suppress_warnings: Suppress ConvergenceWarning display (default: True)
            track_warnings: Track and report warning counts (default: False)
        """
        super().__init__()
        self.n_estimators = n_estimators
        self.max_iter = max_iter
        self.random_state = random_state
        self.verbose = verbose
        self.fallback_method = fallback_method
        self.n_jobs = n_jobs  # 병렬 처리 시 사용할 CPU 코어 수
        self.suppress_warnings = suppress_warnings
        self.track_warnings = track_warnings
        self.warning_counts = {"convergence": 0}  # 경고 카운터

    def _compute_global_features(self, df: pd.DataFrame) -> dict:
        """전역 feature들을 계산합니다 (모든 센서에서 공통으로 사용)."""
        network_avg = df.mean(axis=1, skipna=True)

        df_index = pd.DatetimeIndex(df.index)

        return {
            # 공간적 프록시: 네트워크 평균
            "network_avg": network_avg,
            "network_avg_lag1": network_avg.shift(1).fillna(network_avg),
            "network_avg_lag2": network_avg.shift(2).fillna(network_avg),
            # 시간 특징 (Cyclic encoding)
            "hour_sin": np.sin(2 * np.pi * df_index.hour / 24),
            "hour_cos": np.cos(2 * np.pi * df_index.hour / 24),
            "dayofweek_sin": np.sin(2 * np.pi * df_index.dayofweek / 7),
            "dayofweek_cos": np.cos(2 * np.pi * df_index.dayofweek / 7),
        }

    def _create_feature_matrix(
        self, sensor_data: pd.Series, global_features: dict
    ) -> pd.DataFrame:
        """단일 센서를 위한 feature matrix를 생성합니다."""
        features = pd.DataFrame({sensor_data.name: sensor_data})

        # 전역 features 추가
        for key, value in global_features.items():
            features[key] = value

        # Lag features 추가
        for lag in [1, 2, 3]:
            lag_col = f"{sensor_data.name}_lag{lag}"
            features[lag_col] = sensor_data.shift(lag).fillna(sensor_data)

        return features

    def _impute_sensor(
        self, sensor_data: pd.Series, global_features: dict
    ) -> np.ndarray:
        """단일 센서에 대해 MICE imputation을 수행합니다."""
        imputer = IterativeImputer(
            estimator=ExtraTreesRegressor(
                n_estimators=self.n_estimators,
                random_state=self.random_state,
            ),
            max_iter=self.max_iter,
            random_state=self.random_state,
            verbose=0,  # imputer의 자체 verbose는 꺼두기
        )

        feature_matrix = self._create_feature_matrix(sensor_data, global_features)

        # 경고 추적 with warnings.catch_warnings
        with warnings.catch_warnings(record=True) as w:
            if self.suppress_warnings:
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
            else:
                warnings.filterwarnings("always", category=ConvergenceWarning)

            imputed_array = imputer.fit_transform(feature_matrix)

            # 경고 카운팅
            if self.track_warnings:
                for warning in w:
                    if issubclass(warning.category, ConvergenceWarning):
                        self.warning_counts["convergence"] += 1

        return imputed_array[:, 0]  # 첫 번째 열(원래 센서 데이터)만 반환

    def _apply_fallback(self, df: pd.DataFrame) -> pd.DataFrame:
        """MICE 후 남은 NaN에 대한 fallback 처리를 수행합니다."""
        remaining_nans = df.isna().sum().sum()
        if remaining_nans == 0:
            return df

        if self.verbose > 0:
            print(
                f"Warning: {remaining_nans} NaN values remain after MICE. "
                f"Applying fallback method: {self.fallback_method}"
            )

        # Fallback 전략 적용
        fallback_strategies = {
            "linear": lambda x: x.interpolate(
                method="linear", limit_direction="both", axis=0
            ),
            "ffill": lambda x: x.fillna(method="ffill").fillna(method="bfill"),
            "bfill": lambda x: x.fillna(method="bfill").fillna(method="ffill"),
            "median": lambda x: x.fillna(x.median()),
        }

        df = fallback_strategies.get(
            self.fallback_method, fallback_strategies["linear"]
        )(df)

        # 최종 안전장치
        final_nans = df.isna().sum().sum()
        if final_nans > 0:
            if self.verbose > 0:
                print(
                    f"Warning: {final_nans} NaN values still remain. "
                    f"Filling with 0 as last resort."
                )
            df = df.fillna(0)

        return df

    def _interpolate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        각 센서별로 독립적으로 MICE 보간을 수행합니다.

        Args:
            df: Wide-format DataFrame (index=DatetimeIndex, columns=sensor_ids)

        Returns:
            Interpolated DataFrame with same shape as input
        """
        # 경고 카운터 초기화
        self.warning_counts = {"convergence": 0}

        global_features = self._compute_global_features(df)

        results = Parallel(n_jobs=self.n_jobs, verbose=10)(
            delayed(self._impute_sensor)(df[col], global_features) for col in df.columns
        )

        imputed_data = dict(zip(df.columns, results))
        result_df = pd.DataFrame(imputed_data, index=df.index)

        # 경고 통계 출력
        if self.track_warnings and self.verbose > 0:
            total_sensors = len(df.columns)
            conv_count = self.warning_counts["convergence"]
            print(f"\n[Warning Statistics]")
            print(
                f"  ConvergenceWarning: {conv_count}/{total_sensors} sensors "
                f"({conv_count/total_sensors*100:.1f}%)"
            )

        return self._apply_fallback(result_df)
