from typing import Dict
from songdo_arima.utils import HyperParams
import pickle
import os
from statsmodels.tsa.arima.model import ARIMAResults
from metr.components import TrafficData
import matplotlib.pyplot as plt

def model_test(configs: HyperParams):
    with open(os.path.join(configs.output_root_dir, "results.pkl"), "rb") as f:
        results: Dict[str, Dict[str, str]] = pickle.load(f)
    
    test_data = TrafficData.import_from_hdf(configs.traffic_test_data_path)

    for column, result in results.items():
        model: ARIMAResults = ARIMAResults.load(result["model_path"])
        forecast_result = model.forecast(steps=test_data.data.shape[0])

        plt.figure(figsize=(20, 10))
        plt.plot(test_data.data[column], label="Actual")
        plt.plot(forecast_result, label="Forecast")
        plt.legend()
        plt.title(f"{column} Forecast")
        plt.show()
        # MAE, RMSE 구하기

        break
