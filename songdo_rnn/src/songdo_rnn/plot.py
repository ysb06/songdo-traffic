import seaborn as sns
import glob
import yaml
from collections import defaultdict
import matplotlib.pyplot as plt

PREDICTION_DIR = "./output/predictions"


def plot_loss():
    targets = glob.glob(f"{PREDICTION_DIR}/**/metrics.yaml", recursive=True)
    mae_results = defaultdict(list)
    rmse_results = defaultdict(list)
    smape_results = defaultdict(list)
    for target in targets:
        group = target.split("/")[-3]
        with open(target, "r") as f:
            result = yaml.safe_load(f)
        
        mae_results[group].append(result["test_mae"])
        rmse_results[group].append(result["test_rmse"])
        smape_results[group].append(result["test_smape"])
    
    plot(mae_results)
    plot(rmse_results)
    plot(smape_results)

def plot(results: defaultdict):
    for group, values in results.items():
        avg = sum(values) / len(values)
        sns.barplot(x=[group], y=[avg])
    plt.legend()
    plt.show()
        
        
