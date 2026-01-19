import logging

import numpy as np
import pandas as pd
from pypots.imputation import BRITS
from sklearn.preprocessing import StandardScaler

from .base import Interpolator

logger = logging.getLogger(__name__)


class BRITSInterpolator(Interpolator):
    """
    BRITS (Bidirectional Recurrent Imputation for Time Series) 기반 결측치 보간기.

    Reference:
        Wei Cao, Dong Wang, Jian Li, Hao Zhou, Lei Li, Yitan Li (2018).
        BRITS: Bidirectional Recurrent Imputation for Time Series.
        NeurIPS 2018.

    입력 DataFrame 형식:
        - 인덱스: DatetimeIndex (시간별)
        - 컬럼: 센서 ID
        - 값: 교통량 (float), 결측치는 NaN

    텐서 구조: (N_Samples × N_Steps × N_Features)
    """

    def __init__(
        self,
        n_steps: int = 24,
        rnn_hidden_size: int = 64,
        batch_size: int = 32,
        epochs: int = 50,
        device: str | None = None,
        random_seed: int | None = None,
        fallback_method: str = "linear",
        verbose: int = 0,
    ) -> None:
        """
        Args:
            n_steps: 시퀀스 길이 (기본: 24시간)
            rnn_hidden_size: RNN 히든 레이어 크기
            batch_size: 학습 배치 크기
            epochs: 학습 에폭 수
            learning_rate: 학습률
            device: 학습 디바이스 ('cpu', 'cuda', 'mps' 또는 None=자동 감지)
            random_seed: 재현성을 위한 랜덤 시드
            fallback_method: BRITS 후 남은 NaN에 대한 fallback 방법
                           ('linear', 'ffill', 'bfill', 'median')
            verbose: 로깅 레벨 (0=silent, 1=progress, 2=detailed)
        """
        super().__init__()
        self.name = self.__class__.__name__
        self.n_steps = n_steps
        self.rnn_hidden_size = rnn_hidden_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device if device is not None else self._detect_device()
        self.random_seed = random_seed
        self.fallback_method = fallback_method
        self.verbose = verbose

        self.scaler = StandardScaler()
        self.model = None

    @staticmethod
    def _detect_device() -> str:
        """사용 가능한 디바이스를 자동으로 감지합니다."""
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"

    # ==================== 데이터 변환 함수 ====================

    def _reshape(self, df: pd.DataFrame) -> tuple[np.ndarray, dict]:
        """
        데이터프레임을 BRITS에 맞는 3D 텐서로 변환.

        텐서 구조: (N_Samples × N_Steps × N_Features)

        Args:
            df: 입력 DataFrame (인덱스: datetime, 컬럼: 센서 ID)

        Returns:
            x_3d: 3D numpy 배열 (N_Samples × N_Steps × N_Features)
            meta: 역변환에 필요한 메타데이터
        """
        n_features = df.shape[1]
        total_len = len(df)

        # 스케일링 (신경망 학습 안정성)
        scaled_values = self.scaler.fit_transform(df.values)

        # 샘플 수 계산
        n_samples = total_len // self.n_steps

        if n_samples == 0:
            raise ValueError(
                f"데이터 길이({total_len})가 n_steps({self.n_steps})보다 짧습니다."
            )

        # 유효 길이 계산
        valid_len = n_samples * self.n_steps

        # 자투리 데이터 제외하고 3D로 변환
        truncated_values = scaled_values[:valid_len]
        x_3d = truncated_values.reshape(n_samples, self.n_steps, n_features)

        meta = {
            "columns": df.columns,
            "index": df.index,
            "original_length": total_len,
            "valid_length": valid_len,
            "n_samples": n_samples,
            "n_features": n_features,
            "scaled_remainder": (
                scaled_values[valid_len:] if valid_len < total_len else None
            ),
        }

        return x_3d, meta

    def _inverse_reshape(self, imputed_3d: np.ndarray, meta: dict) -> pd.DataFrame:
        """
        BRITS 결과 텐서를 원래의 데이터프레임 형태로 복원.

        Args:
            imputed_3d: 복원된 3D 텐서 (N_Samples × N_Steps × N_Features)
            meta: reshape에서 저장한 메타데이터

        Returns:
            복원된 DataFrame
        """
        n_features = meta["n_features"]

        # 3D → 2D: (N_Samples × N_Steps × N_Features) → (Valid_Len × N_Features)
        imputed_2d = imputed_3d.reshape(-1, n_features)

        # 자투리 부분 처리
        if meta["scaled_remainder"] is not None:
            imputed_2d = np.vstack([imputed_2d, meta["scaled_remainder"]])

        # 스케일링 복원
        final_values = self.scaler.inverse_transform(imputed_2d)

        # DataFrame 생성
        df_result = pd.DataFrame(
            final_values, index=meta["index"], columns=meta["columns"]
        )

        return df_result

    # ==================== 보간 메인 함수 ====================

    def _apply_fallback(self, df: pd.DataFrame) -> pd.DataFrame:
        """BRITS 후 남은 NaN에 대한 fallback 처리를 수행합니다."""
        remaining_nans = df.isna().sum().sum()
        if remaining_nans == 0:
            return df

        if self.verbose > 0:
            logger.warning(
                f"{remaining_nans} NaN values remain after BRITS. "
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
                logger.warning(
                    f"{final_nans} NaN values still remain. "
                    f"Filling with 0 as last resort."
                )
            df = df.fillna(0)

        return df

    def _interpolate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        BRITS를 사용하여 결측치 보간 수행.

        Args:
            df: 입력 DataFrame (결측치는 NaN)

        Returns:
            결측치가 보간된 DataFrame
        """
        # 1. DataFrame → 3D 텐서 변환
        x_3d, meta = self._reshape(df)

        if self.verbose > 0:
            logger.info(
                f"Tensor shape: {x_3d.shape} "
                f"(samples={meta['n_samples']}, steps={self.n_steps}, features={meta['n_features']})"
            )
            nan_count = np.isnan(x_3d).sum()
            total_count = x_3d.size
            logger.info(
                f"Missing values: {nan_count:,} / {total_count:,} ({nan_count/total_count*100:.2f}%)"
            )
            if meta["valid_length"] < meta["original_length"]:
                remainder = meta["original_length"] - meta["valid_length"]
                logger.info(
                    f"Note: Last {remainder} time steps will be processed by fallback"
                )
            logger.info(
                f"Running BRITS (rnn_hidden_size={self.rnn_hidden_size}, "
                f"epochs={self.epochs}, device={self.device})..."
            )

        # 2. 모델 초기화
        self.model = BRITS(
            n_steps=self.n_steps,
            n_features=meta["n_features"],
            rnn_hidden_size=self.rnn_hidden_size,
            batch_size=self.batch_size,
            epochs=self.epochs,
            device=self.device,
            model_saving_strategy=None,
        )

        # 3. 모델 학습 및 예측
        dataset = {"X": x_3d}
        self.model.fit(dataset)
        predictions = self.model.predict(dataset)

        # 4. 보간된 결과 추출 [N_Samples, N_Steps, N_Features]
        imputed_3d = predictions["imputation"]

        # 5. 3D 텐서 → DataFrame 역변환
        df_imputed = self._inverse_reshape(imputed_3d, meta)

        # 6. 원본에서 결측치였던 부분만 복원값으로 대체
        df_result = df.copy()
        nan_mask = df.isna()
        df_result[nan_mask] = df_imputed[nan_mask]

        # 7. Fallback 처리 (자투리 및 남은 결측치)
        df_result = self._apply_fallback(df_result)

        if self.verbose > 0:
            logger.info("BRITS interpolation completed.")

        return df_result
