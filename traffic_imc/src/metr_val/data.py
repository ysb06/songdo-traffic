"""
데이터 로딩 및 전처리를 위한 유틸리티 모듈
"""
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, Union
from datetime import datetime


def load_metr_imc_data(
    file_path: Union[str, Path],
    key: str = "data"
) -> pd.DataFrame:
    """
    METR-IMC HDF5 데이터 파일을 로드합니다.
    
    Args:
        file_path: HDF5 파일 경로
        key: HDF5 파일 내의 키 (기본값: 'data')
    
    Returns:
        pd.DataFrame: 시간 인덱스와 센서 ID 컬럼을 가진 데이터프레임
                     - Index: DatetimeIndex (시간)
                     - Columns: 센서 ID들
                     - Values: 교통량 데이터 (float64)
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {file_path}")
    
    df = pd.read_hdf(file_path, key=key)
    
    # 인덱스가 DatetimeIndex인지 확인
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("데이터의 인덱스가 DatetimeIndex가 아닙니다.")
    
    # 시간순 정렬 확인
    if not df.index.is_monotonic_increasing:
        df = df.sort_index()
    
    return df


def split_train_test_by_ratio(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    shuffle: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    데이터를 train/test로 비율 기반으로 분할합니다.
    
    Args:
        df: 분할할 데이터프레임
        train_ratio: 훈련 데이터 비율 (0.0 ~ 1.0)
        shuffle: 데이터 셔플 여부 (시계열 데이터의 경우 일반적으로 False)
    
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (train_df, test_df)
    """
    if not 0.0 < train_ratio < 1.0:
        raise ValueError(f"train_ratio는 0과 1 사이여야 합니다: {train_ratio}")
    
    if shuffle:
        df = df.sample(frac=1.0, random_state=42)
    
    split_idx = int(len(df) * train_ratio)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    return train_df, test_df


def split_train_test_by_date(
    df: pd.DataFrame,
    split_date: Union[str, datetime, pd.Timestamp]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    특정 날짜를 기준으로 데이터를 train/test로 분할합니다.
    
    Args:
        df: 분할할 데이터프레임 (DatetimeIndex 필요)
        split_date: 분할 기준 날짜 (이 날짜 이전은 train, 이후는 test)
                   예: '2024-01-01', datetime(2024, 1, 1)
    
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (train_df, test_df)
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("날짜 기반 분할을 위해서는 DatetimeIndex가 필요합니다.")
    
    split_date = pd.Timestamp(split_date)
    
    train_df = df[df.index < split_date]
    test_df = df[df.index >= split_date]
    
    if len(train_df) == 0:
        raise ValueError(f"분할 날짜({split_date})가 너무 이릅니다. 훈련 데이터가 없습니다.")
    if len(test_df) == 0:
        raise ValueError(f"분할 날짜({split_date})가 너무 늦습니다. 테스트 데이터가 없습니다.")
    
    return train_df, test_df


def split_train_val_test(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    shuffle: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    데이터를 train/validation/test로 분할합니다.
    
    Args:
        df: 분할할 데이터프레임
        train_ratio: 훈련 데이터 비율 (0.0 ~ 1.0)
        val_ratio: 검증 데이터 비율 (0.0 ~ 1.0)
        shuffle: 데이터 셔플 여부
    
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: (train_df, val_df, test_df)
    """
    if not 0.0 < train_ratio < 1.0:
        raise ValueError(f"train_ratio는 0과 1 사이여야 합니다: {train_ratio}")
    if not 0.0 < val_ratio < 1.0:
        raise ValueError(f"val_ratio는 0과 1 사이여야 합니다: {val_ratio}")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError(f"train_ratio + val_ratio는 1.0 미만이어야 합니다: {train_ratio + val_ratio}")
    
    if shuffle:
        df = df.sample(frac=1.0, random_state=42)
    
    train_end_idx = int(len(df) * train_ratio)
    val_end_idx = int(len(df) * (train_ratio + val_ratio))
    
    train_df = df.iloc[:train_end_idx]
    val_df = df.iloc[train_end_idx:val_end_idx]
    test_df = df.iloc[val_end_idx:]
    
    return train_df, val_df, test_df


def save_split_data(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: Union[str, Path],
    val_df: Optional[pd.DataFrame] = None,
    prefix: str = "metr-imc"
) -> None:
    """
    분할된 데이터를 HDF5 파일로 저장합니다.
    
    Args:
        train_df: 훈련 데이터
        test_df: 테스트 데이터
        output_dir: 출력 디렉토리
        val_df: 검증 데이터 (선택사항)
        prefix: 파일명 접두사
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 저장
    train_path = output_dir / f"{prefix}_train.h5"
    test_path = output_dir / f"{prefix}_test.h5"
    
    train_df.to_hdf(train_path, key='data', mode='w')
    test_df.to_hdf(test_path, key='data', mode='w')
    
    print(f"✓ 훈련 데이터 저장: {train_path} (shape: {train_df.shape})")
    print(f"✓ 테스트 데이터 저장: {test_path} (shape: {test_df.shape})")
    
    if val_df is not None:
        val_path = output_dir / f"{prefix}_val.h5"
        val_df.to_hdf(val_path, key='data', mode='w')
        print(f"✓ 검증 데이터 저장: {val_path} (shape: {val_df.shape})")


def get_data_summary(df: pd.DataFrame) -> dict:
    """
    데이터프레임의 요약 정보를 반환합니다.
    
    Args:
        df: 데이터프레임
    
    Returns:
        dict: 요약 정보
    """
    return {
        "shape": df.shape,
        "time_range": (df.index.min(), df.index.max()),
        "n_sensors": len(df.columns),
        "n_timestamps": len(df),
        "missing_values": df.isna().sum().sum(),
        "missing_percentage": (df.isna().sum().sum() / df.size) * 100
    }


# 예시 사용법
if __name__ == "__main__":
    # 데이터 로드
    data_path = "./data/selected_small_v1/metr-imc.h5"
    df = load_metr_imc_data(data_path)
    
    print("="*60)
    print("전체 데이터 정보")
    print("="*60)
    summary = get_data_summary(df)
    print(f"Shape: {summary['shape']}")
    print(f"Time range: {summary['time_range'][0]} ~ {summary['time_range'][1]}")
    print(f"Number of sensors: {summary['n_sensors']}")
    print(f"Number of timestamps: {summary['n_timestamps']}")
    print(f"Missing values: {summary['missing_values']} ({summary['missing_percentage']:.2f}%)")
    print()
    
    # 방법 1: 비율 기반 분할 (80:20)
    # print("="*60)
    # print("방법 1: 비율 기반 분할 (80% train, 20% test)")
    # print("="*60)
    # train_df, test_df = split_train_test_by_ratio(df, train_ratio=0.8)
    # print(f"Train shape: {train_df.shape}")
    # print(f"Train time range: {train_df.index.min()} ~ {train_df.index.max()}")
    # print(f"Test shape: {test_df.shape}")
    # print(f"Test time range: {test_df.index.min()} ~ {test_df.index.max()}")
    # print()
    
    # 방법 2: 날짜 기반 분할
    print("="*60)
    print("방법 2: 날짜 기반 분할 (2025-02-01 기준)")
    print("="*60)
    train_df2, test_df2 = split_train_test_by_date(df, split_date='2025-02-01')
    print(f"Train shape: {train_df2.shape}")
    print(f"Train time range: {train_df2.index.min()} ~ {train_df2.index.max()}")
    print(f"Test shape: {test_df2.shape}")
    print(f"Test time range: {test_df2.index.min()} ~ {test_df2.index.max()}")
    print()

    train_df2.to_hdf("./data/selected_small_v1/metr-imc_train.h5", key='data', mode='w')
    test_df2.to_hdf("./data/selected_small_v1/metr-imc_test.h5", key='data', mode='w')
    print("✓ Train/Test 데이터 저장 완료.")
    print()
    
    # 방법 3: Train/Validation/Test 분할 (70:15:15)
    # print("="*60)
    # print("방법 3: Train/Val/Test 분할 (70% / 15% / 15%)")
    # print("="*60)
    # train_df3, val_df3, test_df3 = split_train_val_test(
    #     df, train_ratio=0.7, val_ratio=0.15
    # )
    # print(f"Train shape: {train_df3.shape}")
    # print(f"Train time range: {train_df3.index.min()} ~ {train_df3.index.max()}")
    # print(f"Validation shape: {val_df3.shape}")
    # print(f"Val time range: {val_df3.index.min()} ~ {val_df3.index.max()}")
    # print(f"Test shape: {test_df3.shape}")
    # print(f"Test time range: {test_df3.index.min()} ~ {test_df3.index.max()}")
    # print()
    
    # # 데이터 저장 (선택)
    # # save_split_data(train_df, test_df, output_dir="./output/split_data")
    # # save_split_data(train_df3, test_df3, val_df=val_df3, output_dir="./output/split_data")
    
    # print("="*60)
    # print("완료!")
    # print("="*60)
