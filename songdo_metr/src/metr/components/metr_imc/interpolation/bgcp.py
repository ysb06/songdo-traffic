import pandas as pd
import numpy as np
from numpy.random import multivariate_normal as mvnrnd
from scipy.stats import wishart
from numpy.random import normal as normrnd
from scipy.linalg import khatri_rao as kr_prod
from numpy.linalg import inv as inv
from numpy.linalg import solve as solve
from numpy.linalg import cholesky as cholesky_lower
from scipy.linalg import cholesky as cholesky_upper
from scipy.linalg import solve_triangular as solve_ut
from tqdm import tqdm
from .base import Interpolator


class BGCPInterpolator(Interpolator):
    """
    Bayesian Gaussian CP decomposition (BGCP) 기반 결측치 보간기.
    
    Reference:
        Xinyu Chen, Zhaocheng He, Lijun Sun (2019). 
        A Bayesian tensor decomposition approach for spatiotemporal traffic data imputation.
        Transportation Research Part C: Emerging Technologies, 98: 73-84.
    
    입력 DataFrame 형식:
        - 인덱스: DatetimeIndex (시간별, 1시간 간격)
        - 컬럼: 센서 ID
        - 값: 교통량 (float), 결측치는 NaN
    
    텐서 구조: (센서 m × 날짜 n × 시간대 f=24)
    """
    
    def __init__(
        self, 
        rank: int = 30,
        burn_iter: int = 200,
        gibbs_iter: int = 100,
        random_seed: int | None = None,
        fallback_method: str = "linear",
        verbose: int = 0,
    ) -> None:
        """
        Args:
            rank: CP 분해의 랭크 (잠재 인자 수)
            burn_iter: 번인(burn-in) 반복 횟수
            gibbs_iter: 깁스 샘플링 반복 횟수
            random_seed: 재현성을 위한 랜덤 시드
            fallback_method: BGCP 후 남은 NaN에 대한 fallback 방법
                           ('linear', 'ffill', 'bfill', 'median')
            verbose: 로깅 레벨 (0=silent, 1=progress, 2=detailed)
        """
        self.name = self.__class__.__name__
        self.rank = rank
        self.burn_iter = burn_iter
        self.gibbs_iter = gibbs_iter
        self.random_seed = random_seed
        self.fallback_method = fallback_method
        self.verbose = verbose
    
    # ==================== BGCP 핵심 함수들 ====================
    
    @staticmethod
    def _mvnrnd_pre(mu: np.ndarray, Lambda: np.ndarray) -> np.ndarray:
        """정밀도 행렬을 사용한 다변량 정규분포 샘플링."""
        src = normrnd(size=(mu.shape[0],))
        return solve_ut(
            cholesky_upper(Lambda, overwrite_a=True, check_finite=False),
            src, lower=False, check_finite=False, overwrite_b=True
        ) + mu
    
    @staticmethod
    def _cp_combine(factor: list[np.ndarray]) -> np.ndarray:
        """CP 분해 인자들을 결합하여 텐서 복원."""
        return np.einsum('is, js, ts -> ijt', factor[0], factor[1], factor[2])
    
    @staticmethod
    def _ten2mat(tensor: np.ndarray, mode: int) -> np.ndarray:
        """텐서를 행렬로 언폴딩."""
        return np.reshape(
            np.moveaxis(tensor, mode, 0), 
            (tensor.shape[mode], -1), 
            order='F'
        )
    
    @staticmethod
    def _cov_mat(mat: np.ndarray, mat_bar: np.ndarray) -> np.ndarray:
        """공분산 행렬 계산."""
        mat = mat - mat_bar
        return mat.T @ mat
    
    def _sample_factor(
        self,
        tau_sparse_tensor: np.ndarray,
        tau_ind: np.ndarray,
        factor: list[np.ndarray],
        k: int,
        beta0: float = 1.0
    ) -> np.ndarray:
        """잠재 인자 샘플링."""
        dim, rank = factor[k].shape
        factor_bar = np.mean(factor[k], axis=0)
        temp = dim / (dim + beta0)
        var_mu_hyper = temp * factor_bar
        var_W_hyper = inv(
            np.eye(rank) + 
            self._cov_mat(factor[k], factor_bar) + 
            temp * beta0 * np.outer(factor_bar, factor_bar)
        )
        var_Lambda_hyper = wishart.rvs(df=dim + rank, scale=var_W_hyper)
        var_mu_hyper = self._mvnrnd_pre(var_mu_hyper, (dim + beta0) * var_Lambda_hyper)
        
        idx = list(filter(lambda x: x != k, range(len(factor))))
        var1 = kr_prod(factor[idx[1]], factor[idx[0]]).T
        var2 = kr_prod(var1, var1)
        var3 = (var2 @ self._ten2mat(tau_ind, k).T).reshape([rank, rank, dim]) + var_Lambda_hyper[:, :, np.newaxis]
        var4 = var1 @ self._ten2mat(tau_sparse_tensor, k).T + (var_Lambda_hyper @ var_mu_hyper)[:, np.newaxis]
        
        for i in range(dim):
            factor[k][i, :] = self._mvnrnd_pre(solve(var3[:, :, i], var4[:, i]), var3[:, :, i])
        return factor[k]
    
    @staticmethod
    def _sample_precision_tau(
        sparse_tensor: np.ndarray,
        tensor_hat: np.ndarray,
        ind: np.ndarray
    ) -> float:
        """정밀도 파라미터 tau 샘플링."""
        var_alpha = 1e-6 + 0.5 * np.sum(ind)
        var_beta = 1e-6 + 0.5 * np.sum(((sparse_tensor - tensor_hat) ** 2) * ind)
        return np.random.gamma(var_alpha, 1 / var_beta)
    
    # ==================== 메인 BGCP 알고리즘 ====================
    
    def _bgcp_impute(self, sparse_tensor: np.ndarray) -> np.ndarray:
        """
        BGCP 알고리즘을 사용하여 결측치 복원.
        
        Args:
            sparse_tensor: 결측치가 NaN인 3D 텐서 (센서 × 날짜 × 시간대)
        
        Returns:
            복원된 텐서
        """
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
        
        dim = np.array(sparse_tensor.shape)
        rank = self.rank
        
        # 결측치 위치 파악 (NaN 처리)
        ind = ~np.isnan(sparse_tensor)
        sparse_tensor_filled = sparse_tensor.copy()
        sparse_tensor_filled[np.isnan(sparse_tensor_filled)] = 0
        
        # 인자 초기화
        factor = [0.1 * np.random.randn(dim[k], rank) for k in range(len(dim))]
        
        tau = 1.0
        factor_plus = [np.zeros((dim[k], rank)) for k in range(len(dim))]
        tensor_hat_plus = np.zeros(dim)
        
        total_iter = self.burn_iter + self.gibbs_iter
        
        # tqdm 진행 바 설정
        iterator = tqdm(
            range(total_iter),
            desc="BGCP Sampling",
            unit="iter",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        )
        
        for it in iterator:
            tau_ind = tau * ind
            tau_sparse_tensor = tau * sparse_tensor_filled
            
            # 각 모드에 대해 인자 샘플링
            for k in range(len(dim)):
                factor[k] = self._sample_factor(tau_sparse_tensor, tau_ind, factor, k)
            
            tensor_hat = self._cp_combine(factor)
            tau = self._sample_precision_tau(sparse_tensor_filled, tensor_hat, ind)
            
            # 번인 이후 샘플 누적
            if it + 1 > self.burn_iter:
                factor_plus = [factor_plus[k] + factor[k] for k in range(len(dim))]
                tensor_hat_plus += tensor_hat
            
            # tqdm 상태 업데이트
            phase = "Burn-in" if it < self.burn_iter else "Gibbs"
            iterator.set_postfix(phase=phase, tau=f"{tau:.4f}")
        
        # 깁스 샘플 평균
        tensor_hat = tensor_hat_plus / self.gibbs_iter
        
        return tensor_hat
    
    # ==================== 데이터 변환 함수 ====================
    
    def _reshape(self, df: pd.DataFrame) -> tuple[np.ndarray, dict]:
        """
        데이터프레임을 BGCP에 맞는 3D 텐서로 변환.
        
        텐서 구조: (센서 m × 날짜 n × 시간대 f=24)
        
        Args:
            df: 입력 DataFrame (인덱스: datetime, 컬럼: 센서 ID)
        
        Returns:
            tensor: 3D numpy 배열 (m × n × 24)
            meta: 역변환에 필요한 메타데이터
        """
        # 시간대 수 (1시간 간격이므로 24)
        hours_per_day = 24
        
        # 센서 수, 전체 시간 수
        num_sensors = len(df.columns)
        num_hours = len(df)
        
        # 날짜 수 계산 (불완전한 날짜는 제외)
        num_days = num_hours // hours_per_day
        valid_hours = num_days * hours_per_day
        
        # 불완전한 마지막 날 데이터 제거
        df_trimmed = df.iloc[:valid_hours]
        
        # DataFrame → 2D 배열 (시간 × 센서) → 전치 (센서 × 시간)
        mat = df_trimmed.values.T  # (센서 × 시간)
        
        # 3D 텐서로 변환: (센서 m × 날짜 n × 시간대 f=24)
        tensor = mat.reshape(num_sensors, num_days, hours_per_day)
        
        # 메타데이터 저장 (역변환용)
        meta = {
            'columns': df.columns,
            'index': df_trimmed.index,
            'original_index': df.index,
            'original_length': len(df),
            'trimmed_length': valid_hours,
            'num_sensors': num_sensors,
            'num_days': num_days,
        }
        
        return tensor, meta
    
    def _inverse_reshape(
        self, 
        tensor: np.ndarray, 
        meta: dict
    ) -> pd.DataFrame:
        """
        BGCP 결과 텐서를 원래의 데이터프레임 형태로 복원.
        
        Args:
            tensor: 복원된 3D 텐서 (센서 × 날짜 × 시간대)
            meta: reshape에서 저장한 메타데이터
        
        Returns:
            복원된 DataFrame
        """
        num_sensors = meta['num_sensors']
        num_days = meta['num_days']
        hours_per_day = 24
        
        # 3D → 2D: (센서 × 날짜 × 24) → (센서 × 시간)
        mat = tensor.reshape(num_sensors, num_days * hours_per_day)
        
        # 전치: (센서 × 시간) → (시간 × 센서)
        mat = mat.T
        
        # DataFrame 생성
        df_result = pd.DataFrame(
            mat,
            index=meta['index'],
            columns=meta['columns']
        )
        
        return df_result
    
    # ==================== 보간 메인 함수 ====================
    
    def _apply_fallback(self, df: pd.DataFrame) -> pd.DataFrame:
        """BGCP 후 남은 NaN에 대한 fallback 처리를 수행합니다."""
        remaining_nans = df.isna().sum().sum()
        if remaining_nans == 0:
            return df

        if self.verbose > 0:
            print(
                f"Warning: {remaining_nans} NaN values remain after BGCP. "
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
        BGCP를 사용하여 결측치 보간 수행.
        
        Args:
            df: 입력 DataFrame (결측치는 NaN)
        
        Returns:
            결측치가 보간된 DataFrame
        """
        original_length = len(df)
        
        # 1. DataFrame → 3D 텐서 변환
        sparse_tensor, meta = self._reshape(df)
        
        if self.verbose > 0:
            print(f"Tensor shape: {sparse_tensor.shape} "
                  f"(sensors={meta['num_sensors']}, days={meta['num_days']}, hours=24)")
            nan_count = np.isnan(sparse_tensor).sum()
            total_count = sparse_tensor.size
            print(f"Missing values: {nan_count:,} / {total_count:,} ({nan_count/total_count*100:.2f}%)")
            if meta['trimmed_length'] < original_length:
                print(f"Note: Last {original_length - meta['trimmed_length']} hours "
                      f"trimmed (not divisible by 24)")
        
        # 2. BGCP로 결측치 복원
        if self.verbose > 0:
            print(f"Running BGCP (rank={self.rank}, burn_iter={self.burn_iter}, "
                  f"gibbs_iter={self.gibbs_iter})...")
        
        imputed_tensor = self._bgcp_impute(sparse_tensor)
        
        # 3. 3D 텐서 → DataFrame 역변환
        df_imputed = self._inverse_reshape(imputed_tensor, meta)
        
        # 4. 원본에서 결측치였던 부분만 복원값으로 대체
        df_result = df.copy()
        trimmed_index = meta['index']
        mask = df.loc[trimmed_index].isna()
        df_result.loc[trimmed_index] = df_result.loc[trimmed_index].where(~mask, df_imputed)
        
        # 5. Fallback 처리 (잘린 부분 및 남은 결측치)
        df_result = self._apply_fallback(df_result)
        
        if self.verbose > 0:
            print("BGCP interpolation completed.")
        
        return df_result