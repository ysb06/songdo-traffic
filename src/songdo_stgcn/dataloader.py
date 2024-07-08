import math
import os

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.utils.data as utils
from sklearn import preprocessing

from .model.dataloader import data_transform, load_adj, load_data
from .model.utility import calc_chebynet_gso, calc_gso


class STGCNDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def setup(self, stage=None):
        adj, self.n_vertex = load_adj(self.config.dataset)
        gso = calc_gso(adj, self.config.gso_type)
        if self.config.graph_conv_type == 'cheb_graph_conv':
            gso = calc_chebynet_gso(gso)
        gso = gso.toarray()
        gso = gso.astype(dtype=np.float32)
        self.config.gso = torch.from_numpy(gso).to(self.config.device)

        dataset_path = './data'
        dataset_path = os.path.join(dataset_path, self.config.dataset)
        data_col = pd.read_csv(os.path.join(dataset_path, 'vel.csv')).shape[0]
        val_and_test_rate = 0.15

        len_val = int(math.floor(data_col * val_and_test_rate))
        len_test = int(math.floor(data_col * val_and_test_rate))
        len_train = int(data_col - len_val - len_test)

        train, val, test = load_data(self.config.dataset, len_train, len_val)
        self.zscore = preprocessing.StandardScaler()
        train = self.zscore.fit_transform(train)
        val = self.zscore.transform(val)
        test = self.zscore.transform(test)

        self.train_x, self.train_y = data_transform(train, self.config.n_his, self.config.n_pred, self.config.device)
        self.val_x, self.val_y = data_transform(val, self.config.n_his, self.config.n_pred, self.config.device)
        self.test_x, self.test_y = data_transform(test, self.config.n_his, self.config.n_pred, self.config.device)

    def train_dataloader(self):
        train_data = utils.TensorDataset(self.train_x, self.train_y)
        return utils.DataLoader(dataset=train_data, batch_size=self.config.batch_size, shuffle=False)

    def val_dataloader(self):
        val_data = utils.TensorDataset(self.val_x, self.val_y)
        return utils.DataLoader(dataset=val_data, batch_size=self.config.batch_size, shuffle=False)

    def test_dataloader(self):
        test_data = utils.TensorDataset(self.test_x, self.test_y)
        return utils.DataLoader(dataset=test_data, batch_size=self.config.batch_size, shuffle=False)