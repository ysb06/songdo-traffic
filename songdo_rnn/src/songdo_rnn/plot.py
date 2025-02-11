from typing import Dict, List, Optional
import pandas as pd
import seaborn as sns
import glob
import yaml
from collections import defaultdict
import matplotlib.pyplot as plt
from .utils import load_results_metrics

PREDICTION_DIR = "./output/predictions"


def plot_loss(prediction_dir: str = PREDICTION_DIR):
    mae_results, rmse_results, smape_results, mape_results = load_results_metrics(
        prediction_dir
    )

    plot_results_metrics(mae_results)
    plot_results_metrics(rmse_results)
    plot_results_metrics(smape_results)
    plot_results_metrics(mape_results)


def plot_results_metrics(results: Dict[str, List[float]], title: Optional[str] = None):
    for group, values in results.items():
        avg = sum(values) / len(values)
        sns.barplot(x=[group], y=[avg])
        print(f"{group}: {avg:.4f}")

    if title is not None:
        plt.title(title)
        
    plt.legend()
    plt.show()


def plot_missing(orig_series: pd.Series, targ_series: pd.Series, title: str):
    sensor_name = orig_series.name
    df_plot = pd.DataFrame(
        {
            "orig": orig_series,
            "proc": targ_series,
        }
    )
    df_plot["orig_missing"] = df_plot["orig"].isna()
    df_plot["new_missing"] = (~df_plot["orig_missing"]) & (df_plot["proc"].isna())

    # 그래프 생성
    fig, ax = plt.subplots(figsize=(14, 4))

    ax.plot(
        df_plot.index,
        df_plot["orig"],
        color="black",
        linewidth=0.8,
        label="Original",
    )

    ax.plot(
        df_plot.index,
        df_plot["proc"],
        color="blue",
        linewidth=0.8,
        label="Processed",
    )

    orig_missing_mask = df_plot["orig_missing"]
    ax.scatter(
        df_plot.index[orig_missing_mask],
        [0] * orig_missing_mask.sum(),
        color="yellow",
        s=7,
        label="Original Missing",
    )

    new_missing_mask = df_plot["new_missing"]
    ax.scatter(
        df_plot.index[new_missing_mask],
        [0] * new_missing_mask.sum(),
        color="red",
        s=10,
        label="New Missing (Outlier)",
    )

    ax.legend(loc="upper right")
    ax.set_title(f"{title if title is not None else 'Missings'} in {sensor_name}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")

    plt.tight_layout()
    plt.show()
    plt.close(fig)
