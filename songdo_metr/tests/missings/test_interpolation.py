import os
from typing import List

from metr.components.metr_imc.traffic_data import TrafficData
from metr.components.metr_imc.interpolation import TimeMeanFillInterpolator

print("test_interpolation started")

def test_time_mean_interpolation(
    traffic_data_list: List[TrafficData],
    traffic_data_filename_list: List[str],
    output_dir: str,
    missing_allow_rate: float,
):
    for traffic_data, filename in zip(traffic_data_list, traffic_data_filename_list):
        data = traffic_data.data
        missing_allow_count = int(data.shape[0] * missing_allow_rate)
        print(filename, ":")
        missing_counts = data.isna().sum()
        good_columns = missing_counts[missing_counts <= missing_allow_count].index
        good_data = data[good_columns]
        print(data.shape, "->", good_data.shape)

        traffic_data.data = good_data
        traffic_data.interpolate(TimeMeanFillInterpolator())
        filepath = os.path.join(output_dir, filename)
        traffic_data.to_hdf(filepath)
        print(f"Interpolated data saved to {filepath}")
