from songdo_arima.utils import HyperParams
from metr.components import TrafficData
import pandas as pd

def test_data(configs: HyperParams):
    raw = TrafficData.import_from_hdf("../output/metr-imc-small/metr-imc-training.h5")
    # raw.start_time = pd.Timestamp("2024-08-18 00:00:00")
    raw.start_time = pd.Timestamp("2024-03-01 00:00:00")

    print(raw.data.columns[raw.data[raw.data >= 8000].any()])
    print(raw.data.columns[raw.data.isna().any()])
