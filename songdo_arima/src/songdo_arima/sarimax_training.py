import os
from songdo_arima.utils import HyperParams
import pandas as pd

def train(config: HyperParams):
    output_dir = config.output_root_dir
    model_output_dir = os.path.join(output_dir, "models")
    result_path = os.path.join(output_dir, "results.pkl")