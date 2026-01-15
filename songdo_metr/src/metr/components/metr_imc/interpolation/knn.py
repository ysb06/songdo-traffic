import pandas as pd
import numpy as np
import logging
from sklearn.impute import KNNImputer
from joblib import Parallel, delayed
from typing import Tuple
from ...adj_mx import AdjacencyMatrix # 위에서 정의한 클래스 가정
from .base import Interpolator

logger = logging.getLogger(__name__)

class SpatialKNNInterpolator(Interpolator):
    def __init__(self, adj_matrix: AdjacencyMatrix, n_spatial_features: int = 10, k_time_neighbors: int = 5, n_jobs: int = -1):
        """
        Args:
            adj_matrix: AdjacencyMatrix 객체 (센서 간 거리 정보 포함)
            n_spatial_features (N): 결측치 보정을 위해 참조할 상위 인접 센서의 개수
            k_time_neighbors (k): 유사한 시간대(Row)를 찾을 때 사용할 이웃의 개수
            n_jobs (int): 병렬 실행 워커 수 (-1=모든 CPU 코어, 1=직렬 실행)
        """
        super().__init__()
        self.adj_matrix = adj_matrix
        self.n_spatial_features = n_spatial_features
        self.k_time_neighbors = k_time_neighbors
        self.n_jobs = n_jobs

    def _get_top_n_neighbors(self, sensor_id: str) -> list:
        """인접 행렬에서 특정 센서와 가장 가까운 상위 N개의 센서 ID를 반환합니다."""
        if sensor_id not in self.adj_matrix.sensor_id_to_idx:
            return []
        
        idx = self.adj_matrix.sensor_id_to_idx[sensor_id]
        # 해당 센서의 인접도 벡터 가져오기
        adj_vector = self.adj_matrix.adj_mx[idx]
        
        # 인접도가 높은 순으로 정렬 (argsort는 오름차순이므로 뒤집음)
        sorted_indices = np.argsort(adj_vector)[::-1]
        
        # 인접도가 0보다 큰 센서만 선택 (자기 자신 및 sparsity로 0이 된 센서 제외)
        # 상위 N개 추출
        top_n_indices = [
            i for i in sorted_indices 
            if adj_vector[i] > 0 and i != idx
        ][:self.n_spatial_features]
        
        # 인덱스를 다시 Sensor ID로 변환
        return [self.adj_matrix.sensor_ids[i] for i in top_n_indices]

    def _impute_single_sensor(self, target_sensor: str, subset_df: pd.DataFrame) -> Tuple[str, np.ndarray]:
        """
        단일 센서의 결측치를 보간합니다 (병렬 실행용).
        
        Args:
            target_sensor: 보간 대상 센서 ID
            subset_df: 대상 센서 + 이웃 센서 + network_avg만 포함된 슬라이스 DataFrame
        """
        # KNN Imputation
        knni = KNNImputer(n_neighbors=self.k_time_neighbors)
        imputed_subset = knni.fit_transform(subset_df)
        
        # 대상 센서의 결과만 반환 (첫 번째 컬럼)
        return target_sensor, imputed_subset[:, 0]

    def _prepare_sensor_subset(self, target_sensor: str, df: pd.DataFrame, network_avg: pd.Series) -> pd.DataFrame:
        """
        단일 센서를 위한 최소한의 데이터 슬라이스를 준비합니다.
        메모리 효율을 위해 필요한 컬럼만 포함합니다.
        """
        neighbor_ids = self._get_top_n_neighbors(target_sensor)
        valid_neighbors = [nid for nid in neighbor_ids if nid in df.columns]
        
        if not valid_neighbors:
            logger.info(
                f"No valid neighbors for sensor {target_sensor}. "
                "Using only network average."
            )
        
        # 대상 센서 + 이웃 센서 + network_avg만 포함
        subset_df = pd.DataFrame({
            target_sensor: df[target_sensor],
            **{nid: df[nid] for nid in valid_neighbors},
            '_network_avg': network_avg
        })
        
        return subset_df

    def _interpolate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        각 센서(Column)별로 상위 N개 인접 센서의 데이터를 Feature로 사용하여
        KNN Imputation을 병렬로 수행합니다.
        """
        # 보정된 데이터를 담을 결과 데이터프레임
        imputed_df = df.copy()
        
        # 결측치가 있는 컬럼(센서)들에 대해서만 처리
        columns_with_nan = df.columns[df.isnull().any()].tolist()
        
        if not columns_with_nan:
            logger.info("No missing values found. Skipping interpolation.")
            return imputed_df

        # 전체 네트워크 평균 (시간대별 교통 패턴을 반영하는 global feature)
        # 모든 센서가 결측인 시간대는 전후 시간대로 선형 보간
        network_avg = df.mean(axis=1, skipna=True).interpolate(method='linear')
        # 선형 보간으로도 채워지지 않는 경우 (시작/끝 부분) 전체 평균으로 대체
        network_avg = network_avg.fillna(network_avg.mean())
        
        logger.info(f"Processing {len(columns_with_nan)} sensors with n_jobs={self.n_jobs}...")
        
        # 메모리 효율을 위해 각 센서별 슬라이스를 미리 준비
        # (전체 df 대신 필요한 컬럼만 워커에 전달)
        sensor_subsets = {
            sensor: self._prepare_sensor_subset(sensor, df, network_avg)
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
        
        logger.info("Interpolation completed.")
        return imputed_df