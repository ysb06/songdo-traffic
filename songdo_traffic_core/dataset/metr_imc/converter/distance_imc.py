from typing import List, Tuple
import geopandas as gpd
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class DistancesImc:
    """distances_imc_2024.csv"""

    def __init__(
        self,
        road_data: gpd.GeoDataFrame,  # 표준노드링크 링크 데이터
        distance_limit: float = 2000,  # m 단위
    ) -> None:
        road_data = road_data.to_crs(epsg=5186)
        self.data = pd.DataFrame([])

    def __generate(self, road_data: gpd.GeoDataFrame, distance_limit: float):
        # 깊이 기반 탐색하고 distance_limit에 도달하면 탐색 중지
        pass

    def to_csv(self, dir_path: str) -> None:
        pass