from pmdarima import ARIMA, auto_arima
from songdo_arima.sarimax_training import BestARIMA
import pickle
import os
import matplotlib.pyplot as plt
import pandas as pd


def test_size():
    with open("../output/1630008100.pkl", "rb") as f:
        result: BestARIMA = pickle.load(f)

    pmd_model = result.best_model
    stats_model = pmd_model.arima_res_

    with open("../output/1630008100_pmd.pkl", "wb") as f:
        pickle.dump(pmd_model, f)

    with open("../output/1630008100_stats.pkl", "wb") as f:
        pickle.dump(stats_model, f)

    stats_model.save("../output/1630008100_stats_saved.pkl")


def test_compare_result():
    with open("../output/1630008100.pkl", "rb") as f:
        result: BestARIMA = pickle.load(f)

    test_raw = pd.read_hdf("../datasets/metr-imc/metr-imc-test.h5")
    test_target_data = test_raw["1630008100"]

    pmd_model = result.best_model
    pred_result: pd.Series = pmd_model.predict(n_periods=len(test_target_data))
    pred_result.index = test_target_data.index

    plt.plot(test_target_data, label="True")
    plt.plot(pred_result, label="Predict")
    plt.legend()

    plt.show()
