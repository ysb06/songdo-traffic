from typing import List, Optional
import geopandas as gpd
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def calculate_mean_coords(geometry):
    x_coords = [point[0] for point in geometry.coords]
    y_coords = [point[1] for point in geometry.coords]
    mean_x = np.mean(x_coords)
    mean_y = np.mean(y_coords)
    return pd.Series([mean_x, mean_y], index=["longitude", "latitude"])


class GraphSensorLocations:
    def __init__(self, road_data: gpd.GeoDataFrame, target_ids: Optional[List[str]] = None) -> None:
        road_data = road_data.to_crs(epsg=4326)

        self.result = pd.DataFrame(columns=["sensor_id", "latitude", "longitude"])
        self.result["sensor_id"] = road_data["LINK_ID"]
        self.result[["longitude", "latitude"]] = road_data["geometry"].apply(
            calculate_mean_coords
        )
        if target_ids is not None:
            self.result = self.result[self.result["sensor_id"].isin(target_ids)]

    def to_csv(self, dir_path: str) -> None:
        logger.info(
            f"Saving sensor locations to {dir_path}/graph_sensor_locations.csv..."
        )
        self.result.to_csv(
            f"{dir_path}/graph_sensor_locations.csv", index_label="index"
        )
        logger.info("Complete")
