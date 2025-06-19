import logging
import os
from glob import glob
from typing import List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
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
    HMHZscoreProcessor,
)

from ppom import (
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
from ppom.prediction_test import (
    aggregate_metrics,
    aggregate_top_bottom_n_sensors,
    run_prediction_test,
)
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

logger.info("Generating Corrupted Test Data...")
srep_test_df = simulate_data_corruption(
    test_true_df,
    training_df,
    random_seed=RANDOM_SEED,
)


# -------------- 처리 시작 -------------- #
# Outlier 처리된 데이터셋(Prediction(pred)에도 사용) 생성
## 기본적으로 Outlier 처리된 데이터셋 생성
def do_data_processing() -> None:
    metadata = Metadata.import_from_hdf(METADATA_RAW_PATH)
    logger.info(f"Metadata loaded: {metadata.data.shape}")
    logger.info("Generating base data for outlier processing")
    base_df = generate_base_data(
        training_df,
        metadata.data,
        output_dir=OUTLIER_PTEST_DATA_DIR,
    )
    ## 기본 베이스 기반으로 Outlier 처리된 데이터셋 생성
    logger.info("Generating outlier-processed data")
    outlier_processors: List[OutlierProcessor] = [
        MonthlyHourlyInSensorZscoreOutlierProcessor(),
        HMHZscoreProcessor(),
        InSensorZscoreOutlierProcessor(),
        WinsorizedOutlierProcessor(),
        TrimmedMeanOutlierProcessor(),
        MADOutlierProcessor(),
    ]
    outlier_processors[0].name = "mhzscore"
    outlier_processors[1].name = "holizscore"
    outlier_processors[2].name = "zscore"
    outlier_processors[3].name = "winsor"
    outlier_processors[4].name = "trimm"
    outlier_processors[5].name = "mad"

    # Outlier 제거
    # Prediction 실험에 사용되는 데이터이기도 함, 따라서 p-test 경로에 저장
    outlier_processed_data = remove_outliers(
        base_df,
        outlier_processors,
        output_dir=OUTLIER_PTEST_DATA_DIR,
    )
    logger.info("Outlier-processed data generated")    

    # outlier_processed_data와 같은 데이터(이어야 함)
    ptest_outlier_rm_data = load_outlier_removed_data(OUTLIER_PTEST_DATA_DIR)
    for srep_raw, name in ptest_outlier_rm_data:
        srep_data = pd.concat([srep_raw, srep_test_df.copy()], axis=0, copy=True)
        filepath = os.path.join(OUTLIER_STEST_DATA_DIR, f"{name}.h5")
        srep_data.to_hdf(filepath, key="data")
    stest_outlier_rm_data = load_outlier_removed_data(OUTLIER_STEST_DATA_DIR)

    interpolators: List[Interpolator] = [
        LinearInterpolator(),
        SplineLinearInterpolator(),
        TimeMeanFillInterpolator(),
        MonthlyMeanFillInterpolator(),
        ShiftFillInterpolator(periods=7 * 24),
    ]
    interpolators[0].name = "linear"
    interpolators[1].name = "spline"
    interpolators[2].name = "time_mean"
    interpolators[3].name = "monthly_mean"
    interpolators[4].name = "week_shift"

    # Interpolation 과정에서 이미 테스트는 이미 종료됨
    logger.info("Interpolating outlier-processed data for simple replacement test")
    stest_data = interpolate(
        stest_outlier_rm_data, interpolators, output_dir=INTERPOLATED_STEST_DATA_DIR
    )

    logger.info("Interpolating outlier-processed data for prediction test")
    # Interpolation 과정 후 예측 모델을 생성하고 성능을 검증해야 함
    ptest_data = interpolate(
        ptest_outlier_rm_data, interpolators, output_dir=INTERPOLATED_PTEST_DATA_DIR
    )


def generate_stest_results() -> None:
    # Simple Replacement Test(srep, stest)
    stest_data_list = glob(os.path.join(INTERPOLATED_STEST_DATA_DIR, "*.h5"))
    stest_data: List[Tuple[pd.DataFrame, str]] = [
        (
            TrafficData.import_from_hdf(filepath).data,
            os.path.basename(filepath).split(".")[0],
        )
        for filepath in stest_data_list
    ]
    logger.info(
        f"Calculating metrics for simple replacement test with {len(stest_data)} datasets..."
    )
    diff_mask = (srep_test_df != test_true_df) | (
        srep_test_df.isna() & ~test_true_df.isna()
    )
    total, mae, rmse, smape = do_simple_replacement_test(stest_data, test_true_df, diff_mask)
    # Save results
    logger.info("Saving results for simple replacement test...")
    total.to_excel(os.path.join(OUTPUT_ROOT_DIR, "stest_total_results.xlsx"))
    mae.to_excel(os.path.join(OUTPUT_ROOT_DIR, "stest_mae_results.xlsx"))
    rmse.to_excel(os.path.join(OUTPUT_ROOT_DIR, "stest_rmse_results.xlsx"))
    smape.to_excel(os.path.join(OUTPUT_ROOT_DIR, "stest_smape_results.xlsx"))


def do_prediction_test() -> None:
    # Prediction Test(pred, ptest)
    ptest_data_list = glob(os.path.join(INTERPOLATED_PTEST_DATA_DIR, "*.h5"))
    ptest_data: List[Tuple[pd.DataFrame, str]] = [
        (
            TrafficData.import_from_hdf(filepath).data,
            os.path.basename(filepath).split(".")[0],
        )
        for filepath in ptest_data_list
    ]
    run_prediction_test(ptest_data, test_true_df, PREDICTION_OUTPUT_DIR)
    aggregate_metrics(PREDICTION_OUTPUT_DIR, only_perfect=False)

    # Time Mean이 가장 좋은 성능을 보인다고 할 때, 수행
    # 현재는 다른 모델을 선정 
    # hzscore_time_mean_data = [
    #     item for item in ptest_data if item[1] == "hzscore-time_mean"
    # ]
    # total_sensors = len(hzscore_time_mean_data[0][0].columns)
    # print(
    #     f"Running prediction test for hzscore-time_mean with all {total_sensors} sensors"
    # )
    # run_prediction_test(
    #     hzscore_time_mean_data, test_true_df, PREDICTION_OUTPUT_DIR, k=total_sensors
    # )
    # aggregate_top_bottom_n_sensors(
    #     PREDICTION_OUTPUT_DIR, target_model_name="hzscore-time_mean", n=5
    # )


def plot_sensor_timeseries(df: pd.DataFrame, target_sensor: str) -> None:
    """
    Plot time series graph for a specific sensor in the dataframe.

    Args:
        raw_df: DataFrame containing traffic data with datetime index
        target_sensor: String identifying the sensor column to plot
    """
    if target_sensor not in df.columns:
        logger.error(f"Sensor {target_sensor} not found in dataframe columns")
        return

    logger.info(f"Plotting time series data for sensor: {target_sensor}")

    plt.figure(figsize=(14, 6))
    plt.plot(df.index, df[target_sensor], linewidth=1)
    plt.title(f"Time Series Data for Sensor: {target_sensor}")
    plt.xlabel("Time")
    plt.ylabel("Traffic Value")
    plt.grid(True, alpha=0.3)

    # Add information about the data range
    non_na_values = df[target_sensor].dropna()
    if len(non_na_values) > 0:
        min_val = non_na_values.min()
        max_val = non_na_values.max()
        mean_val = non_na_values.mean()
        missing_pct = (df[target_sensor].isna().sum() / len(df)) * 100

        info_text = (
            f"Min: {min_val:.2f}, Max: {max_val:.2f}, Mean: {mean_val:.2f}\n"
            f"Missing values: {missing_pct:.2f}%"
        )
        plt.figtext(0.5, 0.01, info_text, ha="center", fontsize=10)

    plt.tight_layout()
    plt.show()

    logger.info(f"Completed plotting time series for sensor: {target_sensor}")


do_data_processing()
generate_stest_results()
do_prediction_test()

# Top 2 Good Sensors
plot_sensor_timeseries(raw_df, "1650042700")
plot_sensor_timeseries(raw_df, "1650043300")

# Top 2 Bad Sensors
plot_sensor_timeseries(raw_df, "1630173301")
plot_sensor_timeseries(raw_df, "1660033202")
