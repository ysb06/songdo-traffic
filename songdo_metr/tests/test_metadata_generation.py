import os

import geopandas as gpd
import pandas as pd

from metr.components.metr_imc.interpolation import TimeMeanFillInterpolator
from metr.components.metr_imc.outlier import HourlyZscoreOutlierProcessor
from metr.subset import MetrSubset
from metr.components.metadata import Metadata

def test_existence(nodelink_dir: str, nodelink_road_data_path: str, target_dir: str):
    print(nodelink_dir, "==>", target_dir)
    assert os.path.exists(nodelink_dir)
    assert os.path.exists(nodelink_road_data_path)
    assert os.path.exists(target_dir)

    # 경로가 모두 존재해야 다음 테스트를 진행할 수 있음

def test_importing(nodelink_dir: str):
    metadata = Metadata.import_from_geopandas(nodelink_dir)
    print(metadata.data)
    print("Shape:", metadata.data.shape)
    print("Columns:", metadata.data.columns)

    # Import 기능 테스트

def test_filtering(nodelink_dir: str, raw_dataset: MetrSubset):
    metadata = Metadata.import_from_geopandas(nodelink_dir)
    taret_ids = raw_dataset.metr_ids
    metadata.sensor_filter = taret_ids
    print(metadata.data)
    print("Shape:", metadata.data.shape)

def test_exporting(nodelink_dir: str, raw_dir: str):
    metadata = Metadata.import_from_geopandas(nodelink_dir)

    hdf_path = os.path.join(raw_dir, "metadata.h5")
    xls_path = os.path.join(raw_dir, "miscellaneous", "metadata.xlsx")

    metadata.to_hdf(hdf_path)
    metadata.to_excel(xls_path)

    assert os.path.exists(hdf_path)
    assert os.path.exists(xls_path)
    

    
