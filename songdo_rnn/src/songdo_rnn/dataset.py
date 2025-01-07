from typing import Optional, Tuple, Union

import lightning as L
import numpy as np
from metr.components import TrafficData
from sklearn.preprocessing import MinMaxScaler
from torch import FloatTensor, Tensor, device
from torch.utils.data import DataLoader, Dataset, Subset


class TrafficDataModule(L.LightningDataModule):
    def __init__(
        self,
        traffic_data_path: str = "../datasets/metr-imc/metr-imc.h5",
        start_datetime: Optional[str] = "2024-03-01 00:00:00",
        end_datetime: Optional[str] = "2024-09-30 23:00:00",
        target_traffic_sensor: int = 0,
        training_data_ratio: float = 0.7,
        validation_data_ratio: float = 0.1,
        seq_length: int = 24,
        batch_size: int = 64,
        num_workers: int = 8,
    ) -> None:
        super().__init__()
        self.traffic_data_path = traffic_data_path
        self.start_datetime = start_datetime
        self.end_datetime = end_datetime
        self.target_traffic_sensor = target_traffic_sensor
        self.training_data_ratio = training_data_ratio
        self.validation_data_ratio = validation_data_ratio
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_dataset: Optional[TrafficDataset] = None
        self.valid_dataset: Optional[TrafficDataset] = None
        self.test_dataset: Optional[TrafficDataset] = None
        self.scaler = None

    def prepare_data(self) -> None:
        """
        Lightning 권장 메서드.
        데이터 다운로드나 공유 자원 초기화가 필요할 경우 사용.
        여기서는 이미 데이터를 h5 파일로 가지고 있다고 가정.
        """
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        traffic_data = TrafficData.import_from_hdf(self.traffic_data_path)
        if self.start_datetime is not None:
            traffic_data.start_time = self.start_datetime
        if self.end_datetime is not None:
            traffic_data.end_time = self.end_datetime

        data = traffic_data.data.iloc[:, self.target_traffic_sensor].values

        dataset = TrafficDataset(data=data, seq_length=self.seq_length)
        self.scaler = dataset.scaler  # scaler 보관

        train_subset, valid_subset, test_subset = split_train_valid_test(
            dataset,
            train_ratio=self.training_data_ratio,
            valid_ratio=self.validation_data_ratio,
        )
        self.train_dataset = train_subset
        self.valid_dataset = valid_subset
        self.test_dataset = test_subset

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            # num_workers=self.num_workers,     # worker를 생성하는 과정에서 속도 저하
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            # num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            # num_workers=self.num_workers,
        )

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            # num_workers=self.num_workers,
        )


class TrafficDataset(Dataset):
    def __init__(
        self,
        data: np.ndarray,
        seq_length: int = 24,
        # 미리 지정된 torch.device 메모리에 올릴 경우 사용
        torch_device: Optional[Union[device, str]] = device("cpu"),
    ):
        super().__init__()
        self.data = data
        self.seq_length = seq_length
        # 유효한 샘플 개수: len(data) - seq_length
        self.valid_length = len(data) - seq_length

        self.device = torch_device

        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.scaler.fit(data.reshape(-1, 1))
        self.scaled_data = self.scaler.transform(data.reshape(-1, 1))
        self.scaled_data = FloatTensor(self.scaled_data).to(self.device)

    def __len__(self) -> int:
        return self.valid_length if self.valid_length > 0 else 0

    def __getitem__(self, index: int):
        x = self.scaled_data[index : index + self.seq_length]
        y = self.scaled_data[index + self.seq_length]

        return x, y


def split_train_test(
    dataset: TrafficDataset, train_ratio: float = 0.8
) -> tuple[Subset, Subset]:
    total_len = len(dataset)
    if total_len <= 0:
        raise ValueError("Dataset length must be greater than 0")

    # 학습셋 크기
    train_len = round(total_len * train_ratio)
    train_len = max(1, min(train_len, total_len - 1))

    # 전체 유효 인덱스 [0, 1, 2, ..., total_len - 1]
    indices = list(range(total_len))

    train_indices = indices[:train_len]
    if len(train_indices) < dataset.seq_length:
        raise ValueError(
            "Test dataset generation failed: train_indices length must be greater than seq_length"
        )
    test_indices = indices[train_len - dataset.seq_length :]

    train_subset = Subset(dataset, train_indices)
    test_subset = Subset(dataset, test_indices)

    return train_subset, test_subset


def split_train_valid_test(
    dataset: TrafficDataset,
    train_ratio: float = 0.8,
    valid_ratio: float = 0.1,
    extend_test: bool = True,
) -> Tuple[Subset, Subset, Subset]:
    total_len = len(dataset)
    train_len = round(total_len * train_ratio)
    valid_len = round(total_len * valid_ratio)
    if train_len < 1 or valid_len < 1:
        raise ValueError("Train and valid dataset length must be greater than 0")

    indices = list(range(total_len))

    train_indices = indices[:train_len]
    valid_indices = indices[train_len : train_len + valid_len]
    test_start = (
        train_len + valid_len - dataset.seq_length
        if extend_test
        else train_len + valid_len
    )
    test_indices = indices[test_start:]

    train_subset = Subset(dataset, train_indices)
    valid_subset = Subset(dataset, valid_indices)
    test_subset = Subset(dataset, test_indices)

    return train_subset, valid_subset, test_subset
