import os

NODELINK_20231218_URL = "https://www.its.go.kr/opendata/nodelinkFileSDownload/DF_193/0"
NODELINK_20240325_URL = "https://www.its.go.kr/opendata/nodelinkFileSDownload/DF_195/0"
NODELINK_DATA_URL = NODELINK_20240325_URL
NODELINK_DEFAULT_DIR = "../datasets/metr-imc/nodelink"
INCHEON_REGION_CODES = ["161", "162", "163", "164", "165", "166", "167", "168", "169"]

os.makedirs(NODELINK_DEFAULT_DIR, exist_ok=True)
