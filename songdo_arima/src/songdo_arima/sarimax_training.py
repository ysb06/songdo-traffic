import os

import pandas as pd
from metr.components import TrafficData

from songdo_arima.utils import HyperParams

import os
import itertools
import warnings
import pickle
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResults
from sklearn.metrics import mean_absolute_error


def train(config: HyperParams):
    output_dir = config.output_root_dir
    model_output_dir = os.path.join(output_dir, "models")
    result_filepath = os.path.join(output_dir, "results.pkl")

    os.makedirs(model_output_dir, exist_ok=True)

    raw = TrafficData.import_from_hdf(config.traffic_training_data_path)
    raw_data = raw.data
    data_size = raw_data.shape[0]
    train_size = int(data_size * config.training_data_ratio)
    train_data = raw_data.iloc[:train_size, :]
    valid_data = raw_data.iloc[train_size:, :]

    target_sensor = raw_data.columns[0]
    train_target = train_data[target_sensor]
    valid_target = valid_data[target_sensor]

    train_sensor(train_target, valid_target)

def train_sensor(train_data: pd.Series, valid_data: pd.Series):
    p = d = q = range(0, 4)
    P = D = Q = range(0, 2)
    s = 24

    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2], s) for x in itertools.product(P, D, Q)]

    best_mae = float('inf')
    best_order = None
    best_seasonal_order = None
    best_model = None

    # warnings.filterwarnings("ignore")  # Ignore convergence warnings

    for order in pdq:
        for seasonal_order in seasonal_pdq:
            print(f"SARIMAX{order}x{seasonal_order} - Fitting...")
            try:
                model = SARIMAX(
                    train_data,
                    order=order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                model_fit: SARIMAXResults = model.fit()
                forecast = model_fit.forecast(steps=len(valid_data))
                mae = mean_absolute_error(valid_data, forecast)
                if mae < best_mae:
                    best_mae = mae
                    best_order = order
                    best_seasonal_order = seasonal_order
                    best_model = model_fit
            except Exception as e:
                print(f"Error fitting SARIMAX{order}x{seasonal_order}: {e}")
                continue

    print(f'Best SARIMAX{best_order}x{best_seasonal_order} - MAE: {best_mae}')

    return best_model