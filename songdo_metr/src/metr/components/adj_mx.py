import os
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import logging
import pickle
from tqdm import tqdm

logger = logging.getLogger(__name__)


class AdjacencyMatrix:
    def __init__(
        self, data: Optional[Tuple[List[str], Dict[str, str], np.ndarray]] = None
    ) -> None:
        self.data = data

    @property
    def sensor_ids(self) -> List[str]:
        return self.data[0]

    @property
    def sensor_id_to_idx(self) -> Dict[str, str]:
        return self.data[1]

    @property
    def adj_mx(self) -> np.ndarray:
        return self.data[2]

    @property
    def data_exists(self) -> bool:
        return self.data is not None

    def to_pickle(self, dir_path: str, filename: str = "adj_mx.pkl") -> None:
        file_path = os.path.join(dir_path, filename)
        logger.info(f"Saving data to {file_path}...")
        with open(file_path, "wb") as f:
            pickle.dump(self.data, f)
        logger.info("Saving Complete")


def import_adj_mx(filepath: str) -> AdjacencyMatrix:
    with open(filepath, "rb") as f:
        data = pickle.load(f)
    adj_mx_data = AdjacencyMatrix(data)
    return adj_mx_data
