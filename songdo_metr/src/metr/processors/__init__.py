RAW_DATASET_ROOT_DIR = "../datasets/metr-imc"
RAW_MISCELLANEOUS_DIR = "../datasets/metr-imc/miscellaneous"

# For Node-Link Raw Data
NODELINK_RAW_URL = (
    "https://www.its.go.kr/opendata/nodelinkFileSDownload/DF_195/0"  # 2024-03-25
)
NODELINK_TARGET_DIR = "../datasets/metr-imc/nodelink"
NODELINK_LINK_FILENAME = "imc_link.shp"
NODELINK_TURN_FILENAME = "imc_turninfo.dbf"

# For IMCRTS Raw Data
IMCRTS_DIR = "../datasets/metr-imc/imcrts"
IMCRTS_FILENAME = "imcrts_data.pkl"
IMCRTS_START_DATE = "20230901"
IMCRTS_END_DATE = "20240930"
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