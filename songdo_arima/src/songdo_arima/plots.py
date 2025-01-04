import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import yaml
import textwrap

SARIMAX_RESULTS_DIR = "../output"

def plot_results():
    print("Crawling from", os.path.abspath(SARIMAX_RESULTS_DIR))
    print("Found:", os.path.exists(SARIMAX_RESULTS_DIR))
    targets = glob.glob(f"{SARIMAX_RESULTS_DIR}/**/results.yaml", recursive=True)
    
    mae = {}
    rmse = {}
    for target in targets:
        temp = target.split("/")
        name = temp[-4] + "/" + temp[-3]
        with open(target, "r") as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
            mae[name] = data["mean_MAE"]
            rmse[name] = data["mean_RMSE"]
    
    x = np.arange(len(mae))
    plt.figure(figsize=(10, 5))
    plt.bar(x - 0.2, mae.values(), width=0.4, label="MAE")
    plt.bar(x + 0.2, rmse.values(), width=0.4, label="RMSE")
    wrapped_labels = [textwrap.fill(label, width=16) for label in mae.keys()]
    plt.xticks(x, wrapped_labels, ha='right')
    plt.legend()
    plt.show()
        

if __name__ == "__main__":
    plot_results()