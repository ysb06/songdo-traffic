import argparse
import logging
import os
from copy import deepcopy
from glob import glob
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from metr.components import Metadata, TrafficData
from metr.components.metr_imc.interpolation import (
    Interpolator,
    LinearInterpolator,
    MonthlyMeanFillInterpolator,
    ShiftFillInterpolator,
    SplineLinearInterpolator,
    TimeMeanFillInterpolator,
)
from metr.components.metr_imc.outlier import (
    HourlyInSensorZscoreOutlierProcessor,
    InSensorZscoreOutlierProcessor,
    MADOutlierProcessor,
    MonthlyHourlyInSensorZscoreOutlierProcessor,
    OutlierProcessor,
    TrimmedMeanOutlierProcessor,
    WinsorizedOutlierProcessor,
)
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from tqdm import tqdm

# from songdo_rnn.preprocessing.missing import interpolate
# from songdo_rnn.preprocessing.outlier import remove_outliers
from ppom import (
    BASE_DATA_PATH,
    INTERPOLATED_PTEST_DATA_DIR,
    INTERPOLATED_STEST_DATA_DIR,
    METADATA_RAW_PATH,
    OUTLIER_PTEST_DATA_DIR,
    OUTLIER_STEST_DATA_DIR,
    OUTPUT_ROOT_DIR,
    PREDICTION_OUTPUT_DIR,
    RAW_DATA_PATH,
)
from ppom.missing_processor import interpolate
from ppom.outlier_processor import (
    generate_base_data,
    load_outlier_removed_data,
    remove_outliers,
)
from ppom.prediction_test import aggregate_metrics, do_prediction_test
from ppom.preprocessor import generate_training_test_set, simulate_data_corruption
from ppom.simple_replacement_test import do_simple_replacement_test
from ppom.utils import fix_seed

logging.basicConfig(
    format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
    datefmt="%Y/%m/%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

TRAINING_START_DATETIME = "2022-11-01 00:00:00"
TEST_START_DATETIME = "2024-10-01 00:00:00"
END_DATETIME = "2024-10-31 23:00:00"
RANDOM_SEED = 42

fix_seed(RANDOM_SEED)

raw_data = TrafficData.import_from_hdf(RAW_DATA_PATH)
raw_df = raw_data.data
logger.info(f"Raw data loaded: {raw_df.shape}")

# 학습, 테스트 데이터셋 생성
# 테스트 데이터셋은 이상치와 결측치가 모두 없다는 가정
logger.info("Splitting data into training and test sets")
training_df, test_true_df = generate_training_test_set(
    raw_df,
    training_start_datetime=TRAINING_START_DATETIME,
    test_start_datetime=TEST_START_DATETIME,
    test_end_datetime=END_DATETIME,
    ensure_no_na_in_test=True,
    save_dir=OUTPUT_ROOT_DIR,
)
logger.info(f"Training data shape: {training_df.shape}")
logger.info(f"Test data shape: {test_true_df.shape}")

# -------------- 처리 시작 -------------- #
# Outlier 처리된 데이터셋(Prediction(pred)에도 사용) 생성
## 기본적으로 Outlier 처리된 데이터셋 생성
# metadata = Metadata.import_from_hdf(METADATA_RAW_PATH)
# logger.info(f"Metadata loaded: {metadata.data.shape}")
# logger.info("Generating base data for outlier processing")
# base_df = generate_base_data(
#     training_df,
#     metadata.data,
#     output_dir=OUTLIER_PTEST_DATA_DIR,
# )
# ## 기본 베이스 기반으로 Outlier 처리된 데이터셋 생성
# logger.info("Generating outlier-processed data")
# outlier_processors: List[OutlierProcessor] = [
#     MonthlyHourlyInSensorZscoreOutlierProcessor(),
#     HourlyInSensorZscoreOutlierProcessor(),
#     InSensorZscoreOutlierProcessor(),
#     WinsorizedOutlierProcessor(),
#     TrimmedMeanOutlierProcessor(),
#     MADOutlierProcessor(),
# ]
# outlier_processors[0].name = "mhzscore"
# outlier_processors[1].name = "hzscore"
# outlier_processors[2].name = "zscore"
# outlier_processors[3].name = "winsor"
# outlier_processors[4].name = "trimm"
# outlier_processors[5].name = "mad"

# # Outlier 제거
# # Prediction 실험에 사용되는 데이터이기도 함, 따라서 p-test 경로에 저장
# outlier_processed_data = remove_outliers(
#     base_df,
#     outlier_processors,
#     output_dir=OUTLIER_PTEST_DATA_DIR,
# )
# logger.info("Outlier-processed data generated")

# # Generate data for simple replacement test (srep)
# logger.info("Generating data for simple replacement test")
# srep_test_df = simulate_data_corruption(
#     test_true_df,
#     training_df,
#     random_seed=RANDOM_SEED,
# )

# # outlier_processed_data와 같은 데이터(이어야 함)
# ptest_outlier_rm_data = load_outlier_removed_data(OUTLIER_PTEST_DATA_DIR)
# for srep_raw, name in ptest_outlier_rm_data:
#     srep_data = pd.concat([srep_raw, srep_test_df], axis=0, copy=True)
#     filepath = os.path.join(OUTLIER_STEST_DATA_DIR, f"{name}.h5")
#     srep_data.to_hdf(filepath, key="data")
# logger.info(f"Simple replacement test data shape: {srep_test_df.shape}")
# stest_outlier_rm_data = load_outlier_removed_data(OUTLIER_STEST_DATA_DIR)

# interpolators: List[Interpolator] = [
#     LinearInterpolator(),
#     SplineLinearInterpolator(),
#     TimeMeanFillInterpolator(),
#     MonthlyMeanFillInterpolator(),
#     ShiftFillInterpolator(periods=7 * 24),
# ]
# interpolators[0].name = "linear"
# interpolators[1].name = "spline"
# interpolators[2].name = "time_mean"
# interpolators[3].name = "monthly_mean"
# interpolators[4].name = "week_shift"

# # Interpolation 과정에서 이미 테스트는 이미 종료됨
# logger.info("Interpolating outlier-processed data for simple replacement test")
# stest_data = interpolate(
#     stest_outlier_rm_data, interpolators, output_dir=INTERPOLATED_STEST_DATA_DIR
# )

# logger.info("Interpolating outlier-processed data for prediction test")
# # Interpolation 과정 후 예측 모델을 생성하고 성능을 검증해야 함
# ptest_data = interpolate(
#     ptest_outlier_rm_data, interpolators, output_dir=INTERPOLATED_PTEST_DATA_DIR
# )


# # Simple Replacement Test(srep, stest)
# stest_data_list = glob(os.path.join(INTERPOLATED_STEST_DATA_DIR, "*.h5"))
# stest_data: List[Tuple[pd.DataFrame, str]] = [
#     (
#         TrafficData.import_from_hdf(filepath).data,
#         os.path.basename(filepath).split(".")[0],
#     )
#     for filepath in stest_data_list
# ]
# total, mae, rmse, smape = do_simple_replacement_test(stest_data, test_true_df)


# Prediction Test(pred, ptest)
ptest_data_list = glob(os.path.join(INTERPOLATED_PTEST_DATA_DIR, "*.h5"))
ptest_data: List[Tuple[pd.DataFrame, str]] = [
    (
        TrafficData.import_from_hdf(filepath).data,
        os.path.basename(filepath).split(".")[0],
    )
    for filepath in ptest_data_list
]
do_prediction_test(ptest_data, test_true_df, PREDICTION_OUTPUT_DIR)
aggregate_metrics(PREDICTION_OUTPUT_DIR)
