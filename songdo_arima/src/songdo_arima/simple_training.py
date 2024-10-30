import os
import pickle
from typing import Dict, List, Optional

import pandas as pd
from joblib import Parallel, delayed
from metr.components import TrafficData
from statsmodels.tsa.arima.model import ARIMA, ARIMAResults
from tqdm import tqdm

from songdo_arima.utils import HyperParams


def get_best_model(
    data: pd.Series, p_values: List[int], d_values: List[int], q_values: List[int]
):
    best_aic = float("inf")
    best_order = None
    best_model: Optional[ARIMAResults] = None

    for p in p_values:
        for d in d_values:
            for q in q_values:
                model = ARIMA(data, order=(p, d, q))
                try:
                    model_fit: ARIMAResults = model.fit()
                    aic = model_fit.aic
                    # Todo: AIC 기준으로 Best Model을 선택하지만, 추후 MAE, RMSE 등으로 변경 할 것
                    if aic < best_aic:
                        best_aic = aic
                        best_order = (p, d, q)
                        best_model = model_fit
                except Exception as e:
                    print(
                        f"Error fitting ARIMA(p={p}, d={d}, q={q}) for [{data.name}]: {e}"
                    )
                    continue

    return best_model, {"best_order": best_order, "best_aic": best_aic}


def process_column(
    column: str,
    training_data: pd.DataFrame,
    p_values: List[int],
    d_values: List[int],
    q_values: List[int],
    model_output_dir: str
):
    data = training_data[column]
    best_model, best_result = get_best_model(data, p_values, d_values, q_values)
    
    model_path = os.path.join(model_output_dir, f"{column}.pkl")
    best_model.save(model_path)

    best_result["model_path"] = model_path
    return column, best_result


def train(config: HyperParams):
    output_dir = config.output_root_dir
    model_output_dir = os.path.join(output_dir, "models")
    os.makedirs(model_output_dir, exist_ok=True)
    result_path = os.path.join(output_dir, "results.pkl")

    traffic_training_raw = TrafficData.import_from_hdf(
        config.traffic_training_data_path
    )
    training_data = traffic_training_raw.data
    # Todo: Training and Validation Data로 분할하기

    p_values = range(0, config.p_max)
    d_values = range(0, config.d_max)
    q_values = range(0, config.q_max)

    print(f"Training for data [{training_data.shape}]...")
    results_list = Parallel(n_jobs=-1)(
        delayed(process_column)(
            column,
            training_data,
            p_values,
            d_values,
            q_values,
            model_output_dir
        ) for column in tqdm(training_data.columns)
    )
    
    print("Training done.")
    print("Saving results...")

    results = dict(results_list)
    with open(result_path, "wb") as f:
        pickle.dump(results, f)

    print("Results saved.")