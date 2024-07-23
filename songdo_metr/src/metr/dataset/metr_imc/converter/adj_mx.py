import os
from typing import List
import pandas as pd
import numpy as np
import logging
import pickle
from tqdm import tqdm

logger = logging.getLogger(__name__)


class AdjacencyMatrix:
    """adj_mx.pkl"""

    def __init__(
        self,
        distances: pd.DataFrame,
        sensor_ids: List[str],
    ) -> None:
        self.distance_df = distances
        self.sensor_ids = sensor_ids
        self.adj_mx, self.sendsor_id_to_idx = self.__get_adjacency_matrix(self.distance_df, self.sensor_ids)

    def to_pickle(self, dir_path: str, filename: str = "adj_mx.pkl") -> None:
        packs = []
        packs.append(self.sensor_ids)
        packs.append(self.sendsor_id_to_idx)
        packs.append(self.adj_mx)
        with open(os.path.join(dir_path, filename), "wb") as f:
            pickle.dump(packs, f)

    def __get_adjacency_matrix(
        self, distance_df: pd.DataFrame, sensor_ids: List[str], normalized_k=0.1
    ):
        """
        :param normalized_k: entries that become lower than normalized_k after normalization are set to zero for sparsity.
        :return: adjacency matrix
        """
        num_sensors = len(sensor_ids)
        dist_mx = np.zeros((num_sensors, num_sensors), dtype=np.float32)
        dist_mx[:] = np.inf
        # Builds sensor id to index map.
        sensor_id_to_index = {}
        for i, sensor_id in enumerate(sensor_ids):
            sensor_id_to_index[sensor_id] = i
        # Fills cells in the matrix with distances.
        for row in tqdm(
            distance_df.values, total=len(distance_df), desc="Filling matrix"
        ):
            if row[0] not in sensor_id_to_index or row[1] not in sensor_id_to_index:
                continue
            dist_mx[sensor_id_to_index[row[0]], sensor_id_to_index[row[1]]] = row[2]

        # Calculates the standard deviation as theta.
        distances = dist_mx[~np.isinf(dist_mx)].flatten()
        std = distances.std()
        adj_mx = np.exp(-np.square(dist_mx / std))
        # Make the adjacent matrix symmetric by taking the max.
        # adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])

        # Sets entries that lower than a threshold, i.e., k, to zero for sparsity.
        adj_mx[adj_mx < normalized_k] = 0

        return adj_mx, sensor_id_to_index
