from songdo_arima.sarimax_training import train
from songdo_arima.utils import HyperParams


def test_arima(configs: HyperParams):
    result = train(configs)
    print("-"*50)
    print("MAE Result:", result["mean_MAE"])
    print("RMSE Result:", result["mean_RMSE"])
