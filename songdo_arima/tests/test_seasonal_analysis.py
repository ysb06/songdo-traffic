from songdo_arima.utils import HyperParams
from metr.components import TrafficData
from songdo_arima.seasonal_analysis import plot_seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

def test_plot(traffic_training_data_raw: TrafficData):
    print("Data Shape:", traffic_training_data_raw.data.shape)
    
    sensor_id = traffic_training_data_raw.data.columns[0]

    plot_seasonal_decompose(traffic_training_data_raw.data, sensor_id, 24)
    plot_seasonal_decompose(traffic_training_data_raw.data, sensor_id, 24 * 7)
    plot_seasonal_decompose(traffic_training_data_raw.data, sensor_id, 24 * 30)

def test_acf_pacf(traffic_training_data_raw: TrafficData):
    sensor_id = traffic_training_data_raw.data.columns[0]

    plot_acf(traffic_training_data_raw.data[sensor_id])
    plt.show()
    plot_pacf(traffic_training_data_raw.data[sensor_id])
    plt.show()
