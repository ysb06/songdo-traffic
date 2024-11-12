from songdo_arima.sarimax_training import train
from songdo_arima.utils import HyperParams
import pandas as pd
from metr.components import TrafficData
from pmdarima import ARIMA, auto_arima


def test_arima(configs: HyperParams):
    result = train(configs)
    print("-"*50)
    print("MAE Result:", result["mean_MAE"])
    print("RMSE Result:", result["mean_RMSE"])
