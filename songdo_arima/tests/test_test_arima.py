from songdo_arima.test import model_test
from songdo_arima.utils import HyperParams


def test_arima(configs: HyperParams):
    model_test(configs)
