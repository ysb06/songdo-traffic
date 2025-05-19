from typing import Optional, Union
from metr.components import Metadata, TrafficData
import pandas as pd
import os
import numpy as np
import logging

from tqdm import tqdm

logger = logging.getLogger(__name__)


def generate_training_test_set(
    raw_df: pd.DataFrame,
    training_start_datetime: Union[str, pd.DatetimeIndex],
    test_start_datetime: Union[str, pd.DatetimeIndex],
    test_end_datetime: Union[str, pd.DatetimeIndex],
    ensure_no_na_in_test: bool = True,
    save_dir: Optional[str] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """원본 데이터셋을 기반으로 학습용 데이터셋과 테스트용 데이터셋을 날짜 기준으로 분리합니다. 학습에서 사용되는 검증용은 별도로 분리해야 합니다.

    Args:
        raw_df (pd.DataFrame): 원본 데이터셋
        training_start_datetime (Union[str, pd.DatetimeIndex]): 데이터 시작 지점
        test_start_datetime (Union[str, pd.DatetimeIndex]): 학습용 데이터셋과 테스트용 데이터셋을 나누는 기점
        test_end_datetime (Union[str, pd.DatetimeIndex]): 데이터의 종료 지점
        ensure_no_na_in_test (bool, optional): 테스트 데이터셋에 결측치가 포함될 수 있는지 여부. Defaults to True.
        save_dir (Optional[str], optional): 결과를 파일로 저장할 때 저장될 폴더. Defaults to None. None인 경우 저장하지 않음.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: 학습용 데이터셋과 테스트용 데이터셋 DataFrame
    """
    if isinstance(training_start_datetime, str):
        training_start_datetime = pd.to_datetime(training_start_datetime)
    if isinstance(test_start_datetime, str):
        test_start_datetime = pd.to_datetime(test_start_datetime)
    if isinstance(test_end_datetime, str):
        test_end_datetime = pd.to_datetime(test_end_datetime)

    training_df = raw_df.loc[
        training_start_datetime : test_start_datetime - pd.Timedelta(seconds=1)
    ]
    test_df = raw_df.loc[test_start_datetime:test_end_datetime]

    # 주의: 현재 코드로는 Test 데이터셋에서 결측치는 없는 것이 보장되지만 이상치가 있는지는 보장되지 않음
    # Todo: Test 데이터셋에서 이상치도 제거하는 방법을 찾아야 함
    if ensure_no_na_in_test:
        test_df = test_df.dropna(axis=1)
        # 여기서 test_df가 비워져 버리는 경우가 발생할 수 있으므로 추후 문제 생길 경우 대처하는 코드 작성 필요
    training_df = training_df[test_df.columns]

    if save_dir:
        training_df.to_hdf(os.path.join(save_dir, "training_raw.h5"), key="data")
        test_df.to_hdf(os.path.join(save_dir, "test_true.h5"), key="data")

    return training_df, test_df


def _get_outlier_ratio(ref_data: pd.DataFrame, outlier_threshold: float = 3.0):
    ref_values = ref_data.values.flatten()
    ref_values = ref_values[~np.isnan(ref_values)]

    # 표준화점수 Z-Score 기반으로 Outlier를 예상
    # outlier_threshold 이상의 값을 outlier로 간주
    ref_mean = ref_values.mean()
    ref_std = ref_values.std()
    ref_z = (ref_values - ref_mean) / ref_std

    ref_outlier_indices = np.where(np.abs(ref_z) > outlier_threshold)[0]
    ref_outliers = ref_values[ref_outlier_indices]

    outlier_ratio = len(ref_outlier_indices) / len(ref_values)
    return outlier_ratio, ref_outliers


def _get_missing_ratio(ref_data: pd.DataFrame):
    missing_ratio = ref_data.isna().sum().sum() / ref_data.size
    return missing_ratio


def _insert_missing_blocks(target_data: pd.DataFrame, missing_ratio: float):
    corrupted_data = target_data.copy()

    for col_idx, col in tqdm(
        enumerate(target_data.columns),
        desc="Inserting missing blocks...",
        leave=False,
        total=target_data.shape[1],
    ):
        n_missing_per_sensor = round(target_data[col].count() * missing_ratio)
        available_rows = set(range(target_data.shape[0]))

        # Create missing blocks of various lengths
        missing_lengths = []
        remaining = n_missing_per_sensor

        while remaining > 0:
            length = min(np.random.randint(1, n_missing_per_sensor), remaining)
            missing_lengths.append(length)
            remaining -= length

        # Insert missing blocks
        for length in missing_lengths:
            if len(available_rows) < length:
                break

            potential_starts = [
                row
                for row in available_rows
                if all((row + i) in available_rows for i in range(length))
            ]

            if not potential_starts:
                continue

            start_row = np.random.choice(potential_starts)
            for i in range(length):
                row = start_row + i
                corrupted_data.iloc[row, col_idx] = np.nan
                available_rows.remove(row)

    return corrupted_data


def _insert_outliers(
    target_df: pd.DataFrame,
    outlier_ratio: float,
    ref_outliers: np.ndarray,
):
    corrupted_data = target_df.copy()

    if len(ref_outliers) == 0:
        logger.warning("No outliers found in the outlier reference data.")
        return corrupted_data

    n_outliers = int(corrupted_data.size * outlier_ratio)
    # Find valid positions (non-NaN)
    nan_mask = corrupted_data.isna().to_numpy()
    valid_indices = np.where(~nan_mask.flatten())[0]
    if len(valid_indices) < n_outliers:
        n_outliers = len(valid_indices)

    outlier_positions = np.random.choice(valid_indices, n_outliers, replace=False)
    outlier_values = np.random.choice(ref_outliers, n_outliers)

    # Insert outliers
    for idx, pos in tqdm(
        enumerate(outlier_positions),
        desc="Inserting outliers...",
        leave=False,
        total=n_outliers,
    ):
        row = pos // corrupted_data.shape[1]
        col = pos % corrupted_data.shape[1]
        corrupted_data.iloc[row, col] = outlier_values[idx]

    return corrupted_data


def simulate_data_corruption(
    target_data: pd.DataFrame,
    reference_data: Optional[pd.DataFrame] = None,
    outlier_threshold: float = 3.0,
    random_seed: Optional[int] = None,
) -> pd.DataFrame:
    """대상 데이터에 결측치와 이상치를 삽입하여 데이터 손실을 시뮬레이션합니다. 결측치는 연속된 블록으로 삽입되며, 이상치는 주어진 비율에 따라 삽입됩니다.

    Args:
        target_data (pd.DataFrame): 대상 데이터셋. 주로 테스트 데이터셋을 대상으로 사용.
        reference_data (Optional[pd.DataFrame], optional): 결측치와 이상치를 삽입하는 기준이 될 데이터셋. 주로 학습 데이터셋을 대상으로 사용. Defaults to None.
        outlier_threshold (float, optional): 표준 Z-Score 기반 이상치의 기준이되는 Threshold. Defaults to 3.0.
        random_seed (Optional[int], optional): 무작위 데이터 손실 생성의 난수 시드. Defaults to None.

    Returns:
        pd.DataFrame: 결측치와 이상치가 삽입된 데이터셋
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # Use target data as reference if none provided
    if reference_data is None:
        reference_data = target_data

    logger.info("Simulating data corruption...")
    logger.info("Calculating outlier ratio...")
    outlier_ratio, ref_outliers = _get_outlier_ratio(reference_data, outlier_threshold)
    logger.info("Calculating missing ratio...")
    missing_ratio = _get_missing_ratio(reference_data)
    logger.info("Inserting missing blocks...")
    corrupted_data_by_mvs = _insert_missing_blocks(target_data, missing_ratio)
    logger.info("Inserting outliers...")
    corrupted_data = _insert_outliers(
        corrupted_data_by_mvs,
        outlier_ratio,
        ref_outliers,
    )
    logger.info("Data corruption simulation completed.")
    return corrupted_data
