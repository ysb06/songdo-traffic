import pandas as pd
import torch
from metr.dataloader import MetrDataset
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from songdo_stgcn_trainer.utils import HyperParams, get_config


def test_dataset():
    run_comparison("./configs/base.yaml")
    run_comparison("./configs/metr-1.yaml")


def run_comparison(config_path: str):
    config = get_config(config_path)

    train_loader, val_loader, test_loader = load_and_process_data(config)
    train_loader_metr, val_loader_metr, test_loader_metr = get_metr_dataloaders(config)

    assert compare_loaders(train_loader, train_loader_metr)
    assert compare_loaders(val_loader, val_loader_metr)
    assert compare_loaders(test_loader, test_loader_metr)


def load_and_process_data(config: HyperParams, training_ratio=0.7, val_ratio=0.1):
    df = pd.read_hdf(config.tsfilepath)
    num_samples, _ = df.shape

    # Hyperparameters
    n_his = config.window
    n_pred = config.pred_len
    batch_size = config.batch_size

    len_val = round(num_samples * val_ratio)
    len_train = round(num_samples * training_ratio)
    train = df[:len_train]
    val = df[len_train : len_train + len_val]
    test = df[len_train + len_val :]

    scaler = StandardScaler()
    train = scaler.fit_transform(train)
    val = scaler.transform(val)
    test = scaler.transform(test)

    x_train, y_train = data_transform(train, n_his, n_pred, "cpu")
    x_val, y_val = data_transform(val, n_his, n_pred, "cpu")
    x_test, y_test = data_transform(test, n_his, n_pred, "cpu")

    train_data = TensorDataset(x_train, y_train)
    val_data = TensorDataset(x_val, y_val)
    test_data = TensorDataset(x_test, y_test)

    train_loader = DataLoader(train_data, batch_size=batch_size)
    val_loader = DataLoader(val_data, batch_size=batch_size)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    return train_loader, val_loader, test_loader


def get_metr_dataloaders(config: HyperParams, training_ratio=0.7, val_ratio=0.1):
    dataset = MetrDataset.from_file(config.tsfilepath, config.window, config.pred_len)
    train_subset, val_subset, test_subset = dataset.split(training_ratio, val_ratio)

    train_loader_metr = DataLoader(
        train_subset, batch_size=config.batch_size, collate_fn=MetrDataset.collate_fn
    )
    val_loader_metr = DataLoader(
        val_subset, batch_size=config.batch_size, collate_fn=MetrDataset.collate_fn
    )
    test_loader_metr = DataLoader(
        test_subset, batch_size=config.batch_size, collate_fn=MetrDataset.collate_fn
    )

    return train_loader_metr, val_loader_metr, test_loader_metr


def compare_loaders(loader1, loader2):
    for idx, ((x1, y1), (x2, y2)) in enumerate(zip(loader1, loader2)):
        if not torch.equal(x1, x2) or not torch.equal(y1, y2):
            print(f"Data mismatch found at {idx}!")
            return False
    print("All data matches!")
    return True


def load_data(file_path, len_train, len_val):
    df = pd.read_csv(file_path, header=None).values.astype(float)
    train = df[:len_train]
    val = df[len_train : len_train + len_val]
    test = df[len_train + len_val :]
    return train, val, test


def data_transform(data, n_his, n_pred, device):
    # produce data slices for training and testing
    n_route = data.shape[1]
    l = len(data)
    num = l - n_his - n_pred
    x = np.zeros([num, 1, n_his, n_route])
    y = np.zeros([num, n_route])

    cnt = 0
    for i in range(l - n_his - n_pred):
        head = i
        tail = i + n_his
        x[cnt, :, :, :] = data[head:tail].reshape(1, n_his, n_route)
        y[cnt] = data[tail + n_pred - 1]
        cnt += 1
    return torch.Tensor(x).to(device), torch.Tensor(y).to(device)