import os

RAW_ROOT_DIR_PATH = "../datasets/metr-imc"
RAW_MISCELLANEOUS_DIR_PATH = "../datasets/metr-imc/miscellaneous"
RAW_NODELINK_DIR_PATH = "../datasets/metr-imc/nodelink"
RAW_IMCRTS_DIR_PATH = "../datasets/metr-imc/imcrts"

os.makedirs(RAW_MISCELLANEOUS_DIR_PATH, exist_ok=True)
os.makedirs(RAW_NODELINK_DIR_PATH, exist_ok=True)
os.makedirs(RAW_IMCRTS_DIR_PATH, exist_ok=True)

RAW_NODELINK_LINK_FILENAME = "imc_link.shp"
RAW_NODELINK_TURN_FILENAME = "imc_turninfo.dbf"
RAW_IMCRTS_FILENAME = "imcrts_data.pkl"
RAW_IMCRTS_EXCEL_FILENAME = "imcrts_data.xlsx"
RAW_METR_IMC_FILENAME = "metr-imc.h5"
RAW_METR_IMC_EXCEL_FILENAME = "metr-imc.xlsx"
RAW_SENSOR_IDS_FILENAME = "metr_ids.txt"
RAW_METADATA_FILENAME = "metadata.h5"
RAW_SENSOR_LOCATIONS_FILENAME = "graph_sensor_locations.csv"
RAW_DISTANCES_FILENAME = "distances_imc.csv"
RAW_ADJ_MX_FILENAME = "adj_mx.pkl"

RAW_NODELINK_LINK_PATH = os.path.join(RAW_NODELINK_DIR_PATH, RAW_NODELINK_LINK_FILENAME)
RAW_NODELINK_TURN_PATH = os.path.join(RAW_NODELINK_DIR_PATH, RAW_NODELINK_TURN_FILENAME)
RAW_IMCRTS_PATH = os.path.join(RAW_IMCRTS_DIR_PATH, RAW_IMCRTS_FILENAME)
RAW_IMCRTS_EXCEL_PATH = os.path.join(RAW_IMCRTS_DIR_PATH, RAW_IMCRTS_EXCEL_FILENAME)
RAW_METR_IMC_PATH = os.path.join(RAW_ROOT_DIR_PATH, RAW_METR_IMC_FILENAME)
RAW_METR_IMC_EXCEL_PATH = os.path.join(RAW_MISCELLANEOUS_DIR_PATH, RAW_METR_IMC_EXCEL_FILENAME)
RAW_SENSOR_IDS_PATH = os.path.join(RAW_ROOT_DIR_PATH, RAW_SENSOR_IDS_FILENAME)
RAW_METADATA_PATH = os.path.join(RAW_ROOT_DIR_PATH, RAW_METADATA_FILENAME)
RAW_SENSOR_LOCATIONS_PATH = os.path.join(RAW_ROOT_DIR_PATH, RAW_SENSOR_LOCATIONS_FILENAME)
RAW_DISTANCES_PATH = os.path.join(RAW_ROOT_DIR_PATH, RAW_DISTANCES_FILENAME)
RAW_ADJ_MX_PATH = os.path.join(RAW_ROOT_DIR_PATH, RAW_ADJ_MX_FILENAME)

# Other Settings
PDP_KEY = os.environ.get("PDP_KEY")
NODELINK_RAW_URL = (
    "https://www.its.go.kr/opendata/nodelinkFileSDownload/DF_195/0"  # 2024-03-25
)
TARGET_REGION_CODES = ["161", "162", "163", "164", "165", "166", "167", "168", "169"]


RAW_DATASET_ROOT_DIR = "../datasets/metr-imc"
RAW_MISCELLANEOUS_DIR = "../datasets/metr-imc/miscellaneous"

# For Node-Link Raw Data

NODELINK_TARGET_DIR = "../datasets/metr-imc/nodelink"
NODELINK_LINK_FILENAME = "imc_link.shp"
NODELINK_TURN_FILENAME = "imc_turninfo.dbf"

# For IMCRTS Raw Data
IMCRTS_DIR = "../datasets/metr-imc/imcrts"
IMCRTS_FILENAME = "imcrts_data.pkl"
IMCRTS_START_DATE = "20221101"
IMCRTS_END_DATE = "20250310"
INCHEON_REGION_CODES = ["161", "162", "163", "164", "165", "166", "167", "168", "169"]

# For Splitting Training and Test Data
TRAFFIC_FILENAME = "metr-imc.h5"
TRAFFIC_TRAINING_FILENAME = "metr-imc-training.h5"
TRAFFIC_TEST_FILENAME = "metr-imc-test.h5"

TRAFFIC_EXCEL_FILENAME = "metr-imc.xlsx"
TRAFFIC_TRAINING_EXCEL_FILENAME = "metr-imc-training.xlsx"
TRAFFIC_TEST_EXCEL_FILENAME = "metr-imc-test.xlsx"

TRAINING_START_DATETIME = "20230901 00:00:00"
TRAINING_END_DATETIME = "20240831 23:59:59"
TEST_START_DATETIME = "20240901 00:00:00"
TEST_END_DATETIME = "20240930 23:59:59"

# Other Filenames
SENSOR_IDS_FILENAME = "metr_ids.txt"
METADATA_FILENAME = "metadata.h5"
SENSOR_LOCATIONS_FILENAME = "graph_sensor_locations.csv"
DISTANCES_FILENAME = "distances_imc.csv"
ADJ_MX_FILENAME = "adj_mx.pkl"

# Subset Generation
SUBSET_DATASET_ROOT_DIR = "../datasets/metr-imc-small"
SUBSET_MISCELLANEOUS_DIR = "../datasets/metr-imc-small/miscellaneous"
SUBSET_TARGET_SENSOR_FILENAME = "selected_nodes.shp"

# Outlier Generation
OUTLIER_PROCESSED_DIR = "../output/metr-imc-outlier-processed"

# Interpolation
INTERPOLATION_PROCESSED_DIR = "../output/metr-imc-interpolated"
MISSING_FILENAME = "missing.h5"
