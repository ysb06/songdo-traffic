import logging

import numpy as np
import pandas as pd
import torch
from tsl.data import ImputationDataset
from tsl.data.preprocessing import StandardScaler as TSLScaler
from tsl.engines import Imputer
from tsl.nn.models.stgn import GRINModel

from .base import Interpolator

logger = logging.getLogger(__name__)


class GRINInterpolator(Interpolator):
    """
    GRIN (Graph Recurrent Imputation Network) 기반 결측치 보간기.
    
    Reference:
        Andrea Cini, Ivan Marisca, Cesare Alippi (2022).
        Filling the G_ap_s: Multivariate Time Series Imputation by Graph Neural Networks.
        ICLR 2022.
    
    입력 DataFrame 형식:
        - 인덱스: DatetimeIndex (시간별)
        - 컬럼: 센서 ID
        - 값: 교통량 (float), 결측치는 NaN
    
    인접 행렬(adj) 형식:
        - **가우시안 가중치 행렬 권장** (0-1 binary 행렬보다 훨씬 효과적)
        - 형태: torch.Tensor (N_sensors × N_sensors)
        - 값: W[i,j] = exp(-dist[i,j]^2 / σ^2) (가우시안 커널)
        - 스파스 처리: 임계값 이하는 0으로 설정 권장 (e.g., < 0.1)
        - 예시: AdjacencyMatrix.import_from_components(id_list, distances, normalized_k=0.1)
    
    텐서 구조: (N_Samples × N_Steps × N_Nodes × N_Features)
    """
    
    def __init__(
        self,
        adj: torch.Tensor,
        hidden_size: int = 64,
        ff_size: int = 128,
        n_layers: int = 1,
        embedding_size: int = 32,
        window: int = 24,
        stride: int = 1,
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        device: str | None = None,
        checkpoint_path: str | None = None,
        prediction_loss_weight: float = 1.0,
        random_seed: int | None = None,
        fallback_method: str = "linear",
        verbose: int = 0,
    ) -> None:
        """
        Args:
            adj: 인접 행렬 (Connectivity).
                **가우시안 커널 가중치 행렬 권장**: W[i,j] = exp(-dist[i,j]^2/σ^2)
                0-1 binary 행렬보다 거리 기반 가중치가 효과적입니다.
                예: AdjacencyMatrix.import_from_components().adj_mx
            hidden_size: GRU 셀의 히든 차원 크기
            ff_size: Readout 레이어의 차원 크기
            n_layers: DCRNN 셀의 레이어 수
            embedding_size: 노드 임베딩 차원 크기.
                보통 노드 수($N$)에 대해 log_2(N) ~ sqrt{N} 사이의 값을 권장
            window: 시계열 윈도우 크기 (기본: 24시간)
            stride: 윈도우 슬라이딩 간격
            epochs: 학습 에폭 수
            batch_size: 학습 배치 크기
            learning_rate: 학습률
            device: 학습 디바이스 ('cpu', 'cuda', 'mps' 또는 None=자동 감지)
            checkpoint_path: 사전 학습된 모델 체크포인트 경로
            prediction_loss_weight: 중간 예측값(fwd_pred, bwd_pred) 손실 비중
                                   (학습 안정성 조절용, 기본: 1.0)
            random_seed: 재현성을 위한 랜덤 시드
            fallback_method: GRIN 후 남은 NaN에 대한 fallback 방법
                           ('linear', 'ffill', 'bfill', 'median')
            verbose: 로깅 레벨 (0=silent, 1=progress, 2=detailed)

        embedding_size의 경우 
        """
        super().__init__()
        self.name = self.__class__.__name__
        self.adj = adj
        self.hidden_size = hidden_size
        self.ff_size = ff_size
        self.n_layers = n_layers
        self.embedding_size = embedding_size
        self.window = window
        self.stride = stride
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device if device is not None else self._detect_device()
        self.checkpoint_path = checkpoint_path
        self.prediction_loss_weight = prediction_loss_weight
        self.random_seed = random_seed
        self.fallback_method = fallback_method
        self.verbose = verbose
        
        self.model = None
        self.imputer = None
        self.scaler = None
    
    @staticmethod
    def _detect_device() -> str:
        """사용 가능한 디바이스를 자동으로 감지합니다."""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    
    @staticmethod
    def gaussian_kernel_adj(
        distance_matrix: np.ndarray,
        sigma: float | None = None,
        threshold: float = 0.1,
    ) -> torch.Tensor:
        """
        거리 행렬을 가우시안 커널 가중치 행렬로 변환합니다.
        
        Formula: W[i,j] = exp(-dist[i,j]^2 / σ^2)
        
        Args:
            distance_matrix: 센서 간 거리 행렬 (N × N), 단위: meters
                           np.inf는 연결 없음을 의미
            sigma: 가우시안 표준편차 (None이면 자동 계산: distances.std())
            threshold: 임계값 이하 가중치는 0으로 설정 (스파스 행렬)
        
        Returns:
            가우시안 가중치 인접 행렬 (torch.Tensor)
        
        Example:
            >>> distances = np.array([[0, 1000, 5000], [1000, 0, 3000], [5000, 3000, 0]])
            >>> adj = GRINInterpolator.gaussian_kernel_adj(distances)
            >>> print(adj.shape)  # (3, 3)
        """
        # 무한대를 제외한 거리들로 sigma 계산
        valid_distances = distance_matrix[~np.isinf(distance_matrix)]
        if sigma is None:
            sigma = valid_distances.std()
        
        # 가우시안 커널 변환
        adj_mx = np.exp(-np.square(distance_matrix / sigma))
        
        # 임계값 이하 제거 (스파스화)
        adj_mx[adj_mx < threshold] = 0
        
        # 대각 성분은 1로 설정 (self-loop)
        np.fill_diagonal(adj_mx, 1.0)
        
        return torch.tensor(adj_mx, dtype=torch.float32)
    
    # ==================== 데이터 변환 함수 ====================

    def _reshape(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, dict]:
        """
        데이터프레임을 GRIN에 맞는 형태로 변환.
        
        Args:
            df: 입력 DataFrame (인덱스: datetime, 컬럼: 센서 ID)
        
        Returns:
            values: 2D numpy 배열 (시간 × 센서), NaN 유지
            mask: 관측 마스크 (시간 × 센서), True=관측, False=결측
            meta: 역변환에 필요한 메타데이터
        """
        values = df.values.astype(np.float32)  # (시간 × 센서)
        mask = ~np.isnan(values)  # True=관측된 값
        
        meta = {
            'columns': df.columns,
            'index': df.index,
            'num_times': len(df),
            'num_sensors': len(df.columns),
        }
        
        return values, mask, meta
    
    def _inverse_reshape(
        self,
        imputed_values: np.ndarray,
        meta: dict,
    ) -> pd.DataFrame:
        """
        GRIN 결과를 원래의 데이터프레임 형태로 복원.
        
        Args:
            imputed_values: 복원된 2D 배열 (시간 × 센서)
            meta: reshape에서 저장한 메타데이터
        
        Returns:
            복원된 DataFrame
        """
        df_result = pd.DataFrame(
            imputed_values,
            index=meta['index'],
            columns=meta['columns'],
        )
        return df_result
    
    # ==================== 데이터셋 준비 ====================

    def _prepare_dataset(self, df: pd.DataFrame) -> ImputationDataset:
        """
        DataFrame을 TSL ImputationDataset으로 변환.
        
        **수동 정규화 전략**: tsl의 자동 스케일링 버그를 우회하기 위해
        2D 데이터에 대해 전역 정규화를 수동으로 수행합니다.
        역정규화는 _interpolate에서 수동으로 처리합니다.
        
        Args:
            df: 입력 DataFrame (인덱스: datetime, 컬럼: 센서 ID)
        
        Returns:
            ImputationDataset: GRIN 모델용 데이터셋
        """
        from sklearn.preprocessing import StandardScaler
        
        # 1. 2D 데이터 준비
        values = df.values.astype(np.float32)  # (시간 × 센서)
        
        # 2. NaN 위치 저장 (정규화 전)
        nan_mask_2d = np.isnan(values)
        
        # 3. 전역 정규화 (NaN을 제외한 값들로 fit)
        # 2D 배열을 1D로 flatten하여 정규화
        original_shape = values.shape
        values_flat = values.reshape(-1, 1)
        
        self.scaler = StandardScaler()
        # NaN이 아닌 값들로만 fit
        valid_values = values_flat[~np.isnan(values_flat)].reshape(-1, 1)
        self.scaler.fit(valid_values)
        
        # 전체 데이터 transform (NaN은 그대로 유지됨)
        values_scaled_flat = self.scaler.transform(values_flat)
        values_scaled = values_scaled_flat.reshape(original_shape)
        
        # 4. 모델 입력용으로 NaN을 0으로 채움
        # (eval_mask가 NaN 위치를 기억하므로 모델은 마스크를 통해 결측치 구분 가능)
        values_scaled[nan_mask_2d] = 0.0
        
        # 5. 3D로 변환
        values_scaled = values_scaled[..., np.newaxis]  # (시간 × 센서 × 1)
        
        # 6. 결측치 마스크 생성
        # eval_mask: 평가/복원할 위치 (NaN 위치 → True)
        eval_mask = nan_mask_2d[..., np.newaxis].astype(np.uint8)
        # mask (original_mask): 관측된 값의 위치 (non-NaN 위치 → True)
        mask = (~nan_mask_2d)[..., np.newaxis].astype(np.uint8)
        
        # 7. ImputationDataset 생성 (스케일러 등록 없이 - 수동 처리)
        dataset = ImputationDataset(
            target=values_scaled,
            mask=mask,  # 관측된 값의 위치
            eval_mask=eval_mask,  # 복원할 위치
            connectivity=self.adj,
            window=self.window,
            stride=self.stride,
        )
        
        # 8. original_mask를 데이터셋에 명시적으로 추가
        # tsl의 Imputer가 batch.original_mask를 기대하므로 직접 할당
        import torch
        dataset.original_mask = torch.from_numpy(mask).float()
        
        # tsl의 자동 스케일링 사용 안 함 (이미 수동 적용)
        # 역정규화는 _interpolate에서 수동으로 처리
        
        return dataset
    
    # ==================== Fallback 처리 ====================

    def _apply_fallback(self, df: pd.DataFrame) -> pd.DataFrame:
        """GRIN 후 남은 NaN에 대한 fallback 처리를 수행합니다."""
        remaining_nans = df.isna().sum().sum()
        if remaining_nans == 0:
            return df

        if self.verbose > 0:
            logger.warning(
                f"{remaining_nans} NaN values remain after GRIN. "
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
    
    # ==================== 윈도우 복원 ====================

    def _reconstruct_from_windows(
        self,
        window_outputs: np.ndarray,
        original_length: int,
    ) -> np.ndarray:
        """
        윈도우 기반 예측값을 원래 시퀀스로 복원.
        
        중첩 윈도우의 경우 평균을 사용하여 병합합니다.
        
        Args:
            window_outputs: 윈도우별 출력 (n_samples, window, n_nodes, n_features)
            original_length: 원본 시퀀스 길이
        
        Returns:
            reconstructed: 복원된 시퀀스 (original_length, n_nodes)
        """
        n_samples, window_size, n_nodes, n_features = window_outputs.shape
        
        # 누적 합계 및 카운트 배열
        reconstructed = np.zeros((original_length, n_nodes), dtype=np.float32)
        counts = np.zeros((original_length, n_nodes), dtype=np.float32)
        
        for i in range(n_samples):
            start_idx = i * self.stride
            end_idx = start_idx + window_size
            
            if end_idx > original_length:
                end_idx = original_length
                window_slice = window_outputs[i, :end_idx - start_idx, :, 0]
            else:
                window_slice = window_outputs[i, :, :, 0]
            
            reconstructed[start_idx:end_idx] += window_slice
            counts[start_idx:end_idx] += 1
        
        # 평균 계산 (0으로 나누기 방지)
        counts = np.maximum(counts, 1)
        reconstructed = reconstructed / counts
        
        return reconstructed
    
    # ==================== 보간 메인 함수 ====================

    def _interpolate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        GRIN을 사용하여 결측치 보간 수행.
        
        Args:
            df: 입력 DataFrame (결측치는 NaN)
        
        Returns:
            결측치가 보간된 DataFrame
        """
        if self.random_seed is not None:
            torch.manual_seed(self.random_seed)
            np.random.seed(self.random_seed)
        
        # 1. DataFrame → 변환
        values, mask, meta = self._reshape(df)
        
        if self.verbose > 0:
            logger.info(
                f"Data shape: ({meta['num_times']}, {meta['num_sensors']}) "
                f"(times × sensors)"
            )
            nan_count = (~mask).sum()
            total_count = mask.size
            logger.info(
                f"Missing values: {nan_count:,} / {total_count:,} "
                f"({nan_count/total_count*100:.2f}%)"
            )
        
        # 2. 데이터셋 준비
        dataset = self._prepare_dataset(df)
        
        if self.verbose > 0:
            logger.info(
                f"Running GRIN (hidden_size={self.hidden_size}, "
                f"n_layers={self.n_layers}, window={self.window}, "
                f"device={self.device})..."
            )
        
        # 3. 모델 초기화
        if self.model is None:
            self.model = GRINModel(
                input_size=dataset.n_channels,
                hidden_size=self.hidden_size,
                ff_size=self.ff_size,
                embedding_size=self.embedding_size,
                n_layers=self.n_layers,
                n_nodes=dataset.n_nodes,
            ).to(self.device)
        
        # 4. 체크포인트 로드 또는 학습
        if self.checkpoint_path is not None:
            if self.verbose > 0:
                logger.info(f"Loading checkpoint from {self.checkpoint_path}")
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['state_dict'])
        else:
            # Imputer 엔진 초기화 (loss 계산용)
            if self.imputer is None:
                self.imputer = Imputer(
                    model=self.model,
                    loss_fn=torch.nn.functional.l1_loss,
                    prediction_loss_weight=self.prediction_loss_weight,
                )
            
            if self.verbose > 0:
                logger.info(f"Training GRIN for {self.epochs} epochs...")
            
            # DataLoader 생성
            from tsl.data.batch import Batch
            
            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=True,
                collate_fn=Batch.from_data_list,  # tsl.data.Data 객체를 배치로 묶음
            )
            
            # 수동 학습 루프 (optimizer 최적화 단계 포함)
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
            self.model.train()
            
            from tqdm import tqdm
            epoch_iterator = tqdm(
                range(self.epochs),
                desc="GRIN Training",
                unit="epoch",
                disable=(self.verbose == 0),
            )
            
            for epoch in epoch_iterator:
                epoch_loss = 0.0
                for batch in loader:
                    batch = batch.to(self.device)
                    
                    # original_mask가 배치에 없으면 수동으로 추가
                    if not hasattr(batch, 'original_mask'):
                        batch.original_mask = batch.mask
                    
                    # 그래디언트 초기화
                    optimizer.zero_grad()
                    
                    # 손실 계산 (training_step은 loss만 반환, 최적화는 수행 안함)
                    loss = self.imputer.training_step(batch, batch_idx=0)
                    
                    # 역전파 및 가중치 업데이트
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                
                # tqdm 상태 업데이트
                avg_loss = epoch_loss / len(loader)
                epoch_iterator.set_postfix(loss=f"{avg_loss:.4f}")
                
                if self.verbose > 1 and (epoch + 1) % 10 == 0:
                    logger.info(f"Epoch {epoch + 1}/{self.epochs}, Loss: {avg_loss:.4f}")
        
        # 5. 추론
        self.model.eval()
        
        # Imputer가 없으면 생성 (checkpoint 로드 시)
        if self.imputer is None:
            self.imputer = Imputer(
                model=self.model,
                loss_fn=torch.nn.functional.l1_loss,
            )
        
        # 전체 데이터 추론용 DataLoader
        from tsl.data.batch import Batch
        
        inference_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=len(dataset),
            shuffle=False,
            collate_fn=Batch.from_data_list,  # tsl.data.Data 객체를 배치로 묶음
        )
        
        all_outputs = []
        with torch.no_grad():
            for batch in inference_loader:
                batch = batch.to(self.device)
                
                # original_mask가 배치에 없으면 수동으로 추가
                if not hasattr(batch, 'original_mask'):
                    batch.original_mask = batch.mask
                
                output = self.imputer.predict_step(batch, batch_idx=0)
                y_hat = output['y_hat']  # [batch, window, nodes, channels]
                all_outputs.append(y_hat.cpu().numpy())
        
        # 윈도우 출력 결합
        window_outputs = np.concatenate(all_outputs, axis=0)
        
        # 6. 윈도우에서 원래 시퀀스로 복원
        imputed_values = self._reconstruct_from_windows(
            window_outputs,
            meta['num_times'],
        )
        
        # 7. 수동 역정규화 (전역 스케일 복원)
        original_shape = imputed_values.shape
        imputed_values_flat = imputed_values.reshape(-1, 1)
        imputed_values_original = self.scaler.inverse_transform(imputed_values_flat)
        imputed_values = imputed_values_original.reshape(original_shape).astype(np.float32)
        
        # 8. DataFrame 복원
        df_imputed = self._inverse_reshape(imputed_values, meta)
        
        # 9. 원본에서 결측치였던 부분만 복원값으로 대체
        df_result = df.copy()
        nan_mask = df.isna()
        df_result[nan_mask] = df_imputed[nan_mask]
        
        # 10. Fallback 처리 (남은 결측치)
        df_result = self._apply_fallback(df_result)
        
        if self.verbose > 0:
            logger.info("GRIN interpolation completed.")
        
        return df_result