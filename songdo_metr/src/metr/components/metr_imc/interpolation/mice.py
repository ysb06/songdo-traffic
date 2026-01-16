import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.exceptions import ConvergenceWarning
from tqdm import tqdm
from joblib import Parallel, delayed
from typing import TYPE_CHECKING
import warnings

from .base import Interpolator

if TYPE_CHECKING:
    from ...adj_mx import AdjacencyMatrix


class LegacyMICEInterpolator(Interpolator):
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


class SpatialMICEInterpolator(Interpolator):
    """
    Spatial MICE Interpolator with Adjacency Matrix

    KNN Interpolator 방식을 참조하여, 인접 행렬(AdjacencyMatrix)을 기반으로
    공간적으로 가까운 센서들만 선택하여 MICE imputation을 수행합니다.

    Features:
    - Spatial neighbors from adjacency matrix
    - Temporal patterns (hour-of-day, day-of-week) with cyclic encoding
    - Network-wide average traffic (global feature)
    - Memory-efficient slice-based parallel execution
    """

    def __init__(
        self,
        adj_matrix: "AdjacencyMatrix",
        n_spatial_features: int = 10,
        n_estimators: int = 10,
        max_iter: int = 16,
        random_state: int = 42,
        verbose: int = 0,
        fallback_method: str = "linear",
        n_jobs: int = -2,
        suppress_warnings: bool = True,
    ) -> None:
        """
        Args:
            adj_matrix: AdjacencyMatrix 객체 (센서 간 거리 정보 포함)
            n_spatial_features (N): 결측치 보정을 위해 참조할 상위 인접 센서의 개수
            n_estimators: Number of trees in ExtraTreesRegressor
            max_iter: Maximum MICE iterations
            random_state: Random seed for reproducibility
            verbose: Logging level (0=silent, 1=progress, 2=detailed)
            fallback_method: Fallback interpolation for remaining NaNs
                           ('linear', 'ffill', 'bfill', 'median')
            n_jobs: 병렬 실행 워커 수 (-1=모든 CPU 코어, 1=직렬 실행)
            suppress_warnings: Suppress ConvergenceWarning display (default: True)
        """
        super().__init__()
        self.adj_matrix = adj_matrix
        self.n_spatial_features = n_spatial_features
        self.n_estimators = n_estimators
        self.max_iter = max_iter
        self.random_state = random_state
        self.verbose = verbose
        self.fallback_method = fallback_method
        self.n_jobs = n_jobs
        self.suppress_warnings = suppress_warnings

    def _get_top_n_neighbors(self, sensor_id: str) -> list:
        """인접 행렬에서 특정 센서와 가장 가까운 상위 N개의 센서 ID를 반환합니다."""
        if sensor_id not in self.adj_matrix.sensor_id_to_idx:
            return []

        idx = self.adj_matrix.sensor_id_to_idx[sensor_id]
        adj_vector = self.adj_matrix.adj_mx[idx]

        # 인접도가 높은 순으로 정렬
        sorted_indices = np.argsort(adj_vector)[::-1]

        # 인접도가 0보다 큰 센서만 선택 (자기 자신 및 sparsity로 0이 된 센서 제외)
        top_n_indices = [
            i for i in sorted_indices if adj_vector[i] > 0 and i != idx
        ][: self.n_spatial_features]

        return [self.adj_matrix.sensor_ids[i] for i in top_n_indices]

    def _compute_global_features(self, df: pd.DataFrame) -> dict:
        """전역 feature들을 계산합니다 (모든 센서에서 공통으로 사용)."""
        network_avg = df.mean(axis=1, skipna=True)
        # 결측치 처리
        network_avg = network_avg.interpolate(method="linear").fillna(network_avg.mean())

        df_index = pd.DatetimeIndex(df.index)

        return {
            # 공간적 프록시: 네트워크 평균 및 lag
            "network_avg": network_avg,
            "network_avg_lag1": network_avg.shift(1).fillna(network_avg),
            "network_avg_lag2": network_avg.shift(2).fillna(network_avg),
            # 시간 특징 (Cyclic encoding)
            "hour_sin": np.sin(2 * np.pi * df_index.hour / 24),
            "hour_cos": np.cos(2 * np.pi * df_index.hour / 24),
            "dayofweek_sin": np.sin(2 * np.pi * df_index.dayofweek / 7),
            "dayofweek_cos": np.cos(2 * np.pi * df_index.dayofweek / 7),
        }

    def _prepare_sensor_subset(
        self,
        target_sensor: str,
        df: pd.DataFrame,
        global_features: dict,
    ) -> pd.DataFrame:
        """
        단일 센서를 위한 최소한의 데이터 슬라이스를 준비합니다.
        메모리 효율을 위해 필요한 컬럼만 포함합니다.
        """
        neighbor_ids = self._get_top_n_neighbors(target_sensor)
        valid_neighbors = [nid for nid in neighbor_ids if nid in df.columns]

        if not valid_neighbors and self.verbose > 0:
            print(
                f"No valid neighbors for sensor {target_sensor}. "
                "Using only global features."
            )

        # 대상 센서 + 이웃 센서
        subset_data = {
            target_sensor: df[target_sensor],
            **{nid: df[nid] for nid in valid_neighbors},
        }

        return pd.DataFrame(subset_data)

    def _impute_single_sensor(
        self, target_sensor: str, subset_df: pd.DataFrame
    ) -> tuple:
        """
        단일 센서의 결측치를 MICE로 보간합니다 (병렬 실행용).

        Args:
            target_sensor: 보간 대상 센서 ID
            subset_df: 대상 센서 + 이웃 센서 + global features가 포함된 슬라이스 DataFrame
        """
        imputer = IterativeImputer(
            estimator=ExtraTreesRegressor(
                n_estimators=self.n_estimators,
                random_state=self.random_state,
            ),
            max_iter=self.max_iter,
            random_state=self.random_state,
            verbose=0,
        )

        with warnings.catch_warnings():
            if self.suppress_warnings:
                warnings.filterwarnings("ignore", category=ConvergenceWarning)

            imputed_array = imputer.fit_transform(subset_df)

        # 대상 센서의 결과만 반환 (첫 번째 컬럼)
        return target_sensor, imputed_array[:, 0]

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

        fallback_strategies = {
            "linear": lambda x: x.interpolate(
                method="linear", limit_direction="both", axis=0
            ),
            "ffill": lambda x: x.ffill().bfill(),
            "bfill": lambda x: x.bfill().ffill(),
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
        각 센서(Column)별로 상위 N개 인접 센서의 데이터를 Feature로 사용하여
        MICE Imputation을 병렬로 수행합니다.
        """
        imputed_df = df.copy()

        # 결측치가 있는 컬럼(센서)들에 대해서만 처리
        columns_with_nan = df.columns[df.isnull().any()].tolist()

        if not columns_with_nan:
            if self.verbose > 0:
                print("No missing values found. Skipping interpolation.")
            return imputed_df

        # Global features 계산
        global_features = self._compute_global_features(df)

        if self.verbose > 0:
            print(
                f"Processing {len(columns_with_nan)} sensors with n_jobs={self.n_jobs}..."
            )

        # 메모리 효율을 위해 각 센서별 슬라이스를 미리 준비
        sensor_subsets = {
            sensor: self._prepare_sensor_subset(sensor, df, global_features)
            for sensor in columns_with_nan
        }

        # joblib.Parallel로 병렬 실행 (슬라이스만 전달하여 메모리 절약)
        results = Parallel(n_jobs=self.n_jobs, verbose=10)(
            delayed(self._impute_single_sensor)(sensor, sensor_subsets[sensor])
            for sensor in columns_with_nan
        )

        # 결과를 DataFrame에 반영
        for sensor_id, imputed_values in results:
            imputed_df[sensor_id] = imputed_values

        if self.verbose > 0:
            print("Interpolation completed.")

        return self._apply_fallback(imputed_df)
