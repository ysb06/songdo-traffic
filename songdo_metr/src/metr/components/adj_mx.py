import os
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import logging
import pickle

logger = logging.getLogger(__name__)


class AdjacencyMatrix:
    @staticmethod
    def import_from_pickle(filepath: str) -> "AdjacencyMatrix":
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        return AdjacencyMatrix(data)

    def __init__(self, raw: Tuple[List[str], Dict[str, int], np.ndarray]) -> None:
        self.raw = raw

    @property
    def sensor_ids(self) -> List[str]:
        return self.raw[0]

    @property
    def sensor_id_to_idx(self) -> Dict[str, int]:
        return self.raw[1]

    @property
    def adj_mx(self) -> np.ndarray:
        return self.raw[2]

    @property
    def data_exists(self) -> bool:
        return self.raw is not None

    def to_pickle(self, dir_path: str, filename: str = "adj_mx.pkl") -> None:
        file_path = os.path.join(dir_path, filename)
        logger.info(f"Saving data to {file_path}...")
        with open(file_path, "wb") as f:
            pickle.dump(self.raw, f)
        logger.info("Saving Complete")
