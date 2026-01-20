import numpy as np
import pandas as pd
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from tsl.data import ImputationDataset, SpatioTemporalDataModule
from tsl.data.datamodule.splitters import FixedIndicesSplitter
from tsl.data.preprocessing import StandardScaler
from tsl.engines import Imputer
from tsl.metrics.torch import MaskedMAE
from tsl.nn.models import GRINModel
from tsl.utils.casting import torch_to_numpy
from tsl.ops.connectivity import adj_to_edge_index

from .base import Interpolator


# Interpolator 클래스 상속 가정
class GRINInterpolator(Interpolator):
    def __init__(
        self,
        adj,
        hidden_size=64,
        window=24,
        stride=1,
        epochs=100,
        batch_size=32,
        patience=10,
    ):
        super().__init__()
        self.adj = adj
        self.hidden_size = hidden_size
        self.window = window
        self.stride = stride
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience

    def _interpolate(self, df: pd.DataFrame) -> pd.DataFrame:
        # 1. eval_mask 생성 (df의 인덱스와 컬럼을 유지하여 생성)
        # 실제로 값이 존재하는 위치(observed) 확인
        observed_mask = df.notna()
        # 존재하는 값 중 10%를 무작위로 평가용(True)으로 설정
        eval_mask_values = (np.random.rand(*df.shape) < 0.1) & observed_mask.values
        # target=df와 동일한 구조의 DataFrame으로 생성하여 차원 일관성 확보
        eval_mask_df = pd.DataFrame(
            eval_mask_values, index=df.index, columns=df.columns
        )
        eval_mask_3d = eval_mask_values[..., None]
        
        # mask: 관측된 값의 위치 (NaN이 아닌 값)
        mask_3d = observed_mask.values[..., None].astype(np.uint8)
        
        connectivity = adj_to_edge_index(self.adj)

        # 2. tsl ImputationDataset 객체 생성
        # target에 df를 전달하면 index 처리가 자동화됩니다.
        # eval_mask를 통해 모델은 학습 시 이 데이터들을 '보지 못한 결측치'로 간주합니다.
        torch_dataset = ImputationDataset(
            target=df,
            mask=mask_3d,  # 관측 마스크 추가 (Imputer가 original_mask로 사용)
            eval_mask=eval_mask_3d,
            connectivity=connectivity,
            window=self.window,
            stride=self.stride,
        )

        # 3. 데이터 분할 (인자 이름 수정: _index -> _idxs)
        indices = np.arange(len(torch_dataset))
        split_idx = int(len(indices) * 0.8)
        train_val_splitter = FixedIndicesSplitter(
            train_idxs=indices[:split_idx],  #
            val_idxs=indices[split_idx:],
            test_idxs=indices[split_idx:],
        )

        scalers = {"target": StandardScaler(axis=(0, 1))}
        dm = SpatioTemporalDataModule(
            dataset=torch_dataset,
            scalers=scalers,
            splitter=train_val_splitter,
            batch_size=self.batch_size,
        )
        dm.setup()

        # 4. 모델 및 학습 엔진 설정
        model_kwargs = {
            "input_size": torch_dataset.n_channels,
            "hidden_size": self.hidden_size,
            "n_nodes": torch_dataset.n_nodes,
            "n_layers": 1,
            "merge_mode": "mlp",
            "embedding_size": 32,
        }

        imputer = Imputer(
            model_class=GRINModel,
            model_kwargs=model_kwargs,
            optim_class=torch.optim.Adam,
            optim_kwargs={"lr": 0.001},
            loss_fn=MaskedMAE(),  # 마스크된 영역만 계산하는 적절한 지표 사용
        )

        # 5. 학습
        early_stop = EarlyStopping(
            monitor="val_mae", patience=self.patience, mode="min"
        )


        trainer = Trainer(
            max_epochs=self.epochs,
            accelerator="auto",
            devices=1,
            callbacks=[early_stop],
            enable_checkpointing=False,
            logger=False,
        )
        trainer.fit(imputer, datamodule=dm)

        # 6. 전체 데이터 예측
        # get_dataloader()는 인자가 없을 때 전체 데이터셋 로더를 반환합니다.
        output = trainer.predict(imputer, dataloaders=dm.get_dataloader())
        output = imputer.collate_prediction_outputs(output)
        output = torch_to_numpy(output)

        # 7. 결과 복원 및 결합
        y_hat = output["y_hat"].squeeze(-1)
        imputed_df = pd.DataFrame(y_hat, index=df.index, columns=df.columns)

        # 실제 관측값은 유지하고 결측치만 채움
        return df.fillna(imputed_df)
