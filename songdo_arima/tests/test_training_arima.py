from songdo_arima.simple_training import train
from songdo_arima.utils import HyperParams


def test_arima(configs: HyperParams):
    train(configs)
