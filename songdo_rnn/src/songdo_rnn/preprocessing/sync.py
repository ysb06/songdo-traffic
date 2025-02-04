from glob import glob
import os
from typing import List, Set
from metr.components import TrafficData

TARGET_DIR = "./output/missing_processed"
OUTPUT_DIR = "./output/sync_processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def process_sync():
    traffic_files = glob(os.path.join(TARGET_DIR, "*.h5"))

    non_nan_sensors: List[Set[str]] = []
    for file_path in traffic_files:
        traffic_data = TrafficData.import_from_hdf(file_path)
        columns = set(traffic_data.data.columns[traffic_data.data.notna().all()])
        print(f"{len(columns):>5d}", ":", file_path)
        non_nan_sensors.append(columns)
    # traffic_data_idxs의 전체 교집합 구하기
    sync_idx = non_nan_sensors[0]
    for idx in non_nan_sensors[1:]:
        sync_idx = sync_idx.intersection(idx)
    print(f"Sync Index: {len(sync_idx)}")
    
    for file_path in traffic_files:
        traffic_data = TrafficData.import_from_hdf(file_path)
        traffic_data.sensor_filter = list(sync_idx)
        traffic_data.to_hdf(os.path.join(OUTPUT_DIR, os.path.basename(file_path)))
