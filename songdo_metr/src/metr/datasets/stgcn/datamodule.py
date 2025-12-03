from pathlib import Path
from typing import Literal, Optional

import lightning as L
from torch.utils.data import DataLoader

from metr.components.adj_mx import AdjacencyMatrix
from metr.components.metr_imc.traffic_data import TrafficData

from .dataloader import collate_fn
from .dataset import STGCNDataset


class STGCNDataModule(L.LightningDataModule):
    def __init__(
        self,
        dataset_dir_path: str,
        n_his: int = 12,
        n_pred: int = 3,
        batch_size: int = 64,
        num_workers: int = 1,
        shuffle_training: bool = True,
        adj_mx_filename: str = "adj_mx.pkl",
        training_data_filename: str = "metr-imc_train.h5",
        test_data_filename: str = "metr-imc_test.h5",
        train_val_split: float = 0.8,
    ):
        super().__init__()
        self.dataset_dir_path = Path(dataset_dir_path)
        self.adj_mx_path = self.dataset_dir_path / adj_mx_filename
        self.training_data_path = self.dataset_dir_path / training_data_filename
        self.test_data_path = self.dataset_dir_path / test_data_filename
        self.n_his = n_his
        self.n_pred = n_pred
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle_training = shuffle_training
        self.train_val_split = train_val_split

        self.adj_mx_raw: Optional[AdjacencyMatrix] = None
        self.training_dataset: Optional[STGCNDataset] = None
        self.validation_dataset: Optional[STGCNDataset] = None
        self.test_dataset: Optional[STGCNDataset] = None

    def setup(
        self,
        stage: Optional[Literal["fit", "validate", "test", "predict"]] = None,
    ):
        self.adj_mx_raw = AdjacencyMatrix.import_from_pickle(self.adj_mx_path)
        ordered_sensor_ids = self.adj_mx_raw.sensor_ids

        if stage in ["fit", "validate", None]:
            raw = TrafficData.import_from_hdf(self.training_data_path)
            
            # Calculate split date based on train_val_split ratio
            total_length = len(raw.data)
            split_index = int(total_length * self.train_val_split)
            split_date = raw.data.index[split_index]
            
            training_raw, validation_raw = raw.split(split_date)
            training_data_df = training_raw.data
            validation_data_df = validation_raw.data

            training_data_df = training_data_df[ordered_sensor_ids]
            validation_data_df = validation_data_df[ordered_sensor_ids]

            training_data_array = training_data_df.values  # (time_steps, n_vertex)
            validation_data_array = validation_data_df.values  # (time_steps, n_vertex)

            self.training_dataset = STGCNDataset(
                training_data_array,
                self.n_his,
                self.n_pred,
            )
            self.validation_dataset = STGCNDataset(
                validation_data_array,
                self.n_his,
                self.n_pred,
            )

        if stage in ["test", None]:
            test_raw = TrafficData.import_from_hdf(self.test_data_path)
            test_data_df = test_raw.data
            test_data_df = test_data_df[ordered_sensor_ids]
            test_data_array = test_data_df.values  # (time_steps, n_vertex)
            self.test_dataset = STGCNDataset(test_data_array, self.n_his, self.n_pred)

        # Todo: 데이터 스케일링을 어떻게 적용할 것인지 고민 필요, 현재는 데이터 스케일링 적용하지 않음

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.training_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle_training,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.validation_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )
