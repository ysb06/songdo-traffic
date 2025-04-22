import os

OUTPUT_ROOT_DIR = "./output"

OUTLIER_OUTPUT_DIR = os.path.join(OUTPUT_ROOT_DIR, "outlier_processed")
INTERPOLATED_OUTPUT_DIR = os.path.join(OUTPUT_ROOT_DIR, "interpolated")
PREDICTION_OUTPUT_DIR = os.path.join(OUTPUT_ROOT_DIR, "prediction")

OUTLIER_STEST_DATA_DIR = os.path.join(OUTLIER_OUTPUT_DIR, "stest")
OUTLIER_PTEST_DATA_DIR = os.path.join(OUTLIER_OUTPUT_DIR, "ptest")

INTERPOLATED_STEST_DATA_DIR = os.path.join(INTERPOLATED_OUTPUT_DIR, "stest")
INTERPOLATED_PTEST_DATA_DIR = os.path.join(INTERPOLATED_OUTPUT_DIR, "ptest")

RAW_DATA_PATH = "../datasets/metr-imc/metr-imc.h5"
BASE_DATA_PATH = os.path.join(OUTLIER_OUTPUT_DIR, "base.h5")

for dir_path in [
    OUTLIER_OUTPUT_DIR,
    INTERPOLATED_OUTPUT_DIR,
    PREDICTION_OUTPUT_DIR,
    OUTLIER_STEST_DATA_DIR,
    OUTLIER_PTEST_DATA_DIR,
    INTERPOLATED_STEST_DATA_DIR,
    INTERPOLATED_PTEST_DATA_DIR,
]:
    os.makedirs(dir_path, exist_ok=True)