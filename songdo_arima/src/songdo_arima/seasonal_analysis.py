from metr.components import TrafficData
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
import pandas as pd

def plot_seasonal_decompose(df: pd.DataFrame, column: str, period: int):
    decomposition = seasonal_decompose(
        df[column], model="additive", period=period
    )
    _, axes = plt.subplots(4, 1, figsize=(10, 6), sharex=True)

    decomposition.observed.plot(ax=axes[0], legend=False)
    decomposition.trend.plot(ax=axes[1], legend=False)
    decomposition.seasonal.plot(ax=axes[2], legend=False)
    decomposition.resid.plot(ax=axes[3], legend=False)

    axes[0].set_ylabel("Observed")
    axes[1].set_ylabel("Trend")
    axes[2].set_ylabel("Seasonal")
    axes[3].set_ylabel("Residual")

    plt.suptitle(f"{column}")
    plt.xlabel("Date and Time")
    plt.show()
