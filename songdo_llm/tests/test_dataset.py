import random
import pandas as pd
import numpy as np
import pytest
from metr.dataloader import TrafficDataModule as NewDataModule
from tqdm import tqdm
from songdo_llm.dataset import TrafficDataModule as OldDataModule
import torch


def test_loading_data():
    raw_df = pd.read_hdf("../datasets/metr-imc/metr-imc.h5")
    selected_columns = random.sample(list(raw_df.columns), 100)
    print()

    training_df = raw_df.loc[:, list(set(raw_df.columns) - set(selected_columns))]
    test_df = raw_df.loc[:, selected_columns]
    print("Loading...")
    print(training_df.shape)
    print(test_df.shape)

    training_df = training_df[training_df.index < "2024-10-01"]
    test_df = test_df[test_df.index >= "2024-10-01"]
    print("Filtering...")
    print(training_df.shape)
    print(test_df.shape)

    training_df = training_df.dropna(axis="columns", how="all")
    test_df = test_df.dropna(axis="columns", how="all")
    print("Dropping NaN columns...")
    print(training_df.shape)
    print(test_df.shape)

    
    print(training_df)
    print(test_df)


def test_compare_old_and_new_datamodule():
    idx_limit = 10000

    raw_df = pd.read_hdf("../datasets/metr-imc/metr-imc.h5")
    selected_columns = random.sample(list(raw_df.columns), 100)

    training_df = raw_df.loc[:, list(set(raw_df.columns) - set(selected_columns))]
    test_df = raw_df.loc[:, selected_columns]

    training_df = training_df[training_df.index < "2024-10-01"]
    test_df = test_df[test_df.index >= "2024-10-01"]

    training_df = training_df.dropna(axis="columns", how="all")
    test_df = test_df.dropna(axis="columns", how="all")

    # OldDataModule과 NewDataModule을 동일하게 초기화
    
    old_module = OldDataModule(
        training_df=training_df.copy(),
        test_df=test_df.copy(),
        seq_length=24,
        batch_size=8,
        num_workers=0,
    )
    
    new_module = NewDataModule(
        training_df=training_df.copy(),
        test_df=test_df.copy(),
        seq_length=24,
        batch_size=8,
        num_workers=0,
        shuffle_training=False,
    )

    # setup 호출
    print("Setting up old DataModule...")
    old_module.setup("fit")
    print("Setting up new DataModule...")
    new_module.setup("fit")
    print("Setup complete")

    # train_dataloader 비교
    old_train_loader = old_module.train_dataloader()
    new_train_loader = new_module.train_dataloader()

    assert len(old_train_loader) == len(new_train_loader), "Number of data differ"

    for idx, (old_batch, new_batch) in tqdm(enumerate(zip(old_train_loader, new_train_loader)), total=idx_limit):
        assert torch.allclose(old_batch[0], new_batch[0]), "Input data mismatch"

        if idx >= idx_limit:
            break
        

    # 필요하다면 validation/test도 동일 방식으로 비교 가능
