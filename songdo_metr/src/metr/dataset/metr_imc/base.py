import pandas as pd
import os

class MetrDatasetBase:
    def __init__(
        self,
        data_root: str = "./datasets/metr-imc",
        traffic_data_filename: str = "metr-imc.h5",
        ids_filename: str = "metr_ids.txt",
        distances_filename: str = "distances_imc_2023.csv",
        adj_mx_filename: str = "adj_mx.pkl",
    ) -> None:
        self.traffic_df = pd.read_hdf(os.path.join(data_root, traffic_data_filename))
        self.adj_mx = pd.read_pickle(os.path.join(data_root, adj_mx_filename))


class MetrImc:
    def __init__(self) -> None:
        pass