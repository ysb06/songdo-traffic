import os
from typing import List
import pandas as pd

import logging

logger = logging.getLogger(__name__)


import os
from typing import List
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class SensorLocations:
    @staticmethod
    def import_from_csv(filepath: str) -> "SensorLocations":
        raw = pd.read_csv(filepath, index_col="index", dtype={"sensor_id": str})
        return SensorLocations(raw)
    
    def __init__(self, raw: pd.DataFrame) -> None:
        self._raw = raw
        self._sensor_filter = self._raw["sensor_id"].to_list()
        self.data = self._raw.copy()

    @property
    def sensor_filter(self) -> list[str]:
        return self._sensor_filter

    @sensor_filter.setter
    def sensor_filter(self, sensor_ids: list[str]) -> None:
        missing_sensors = set(sensor_ids) - set(self._raw["sensor_id"])
        if missing_sensors:
            logger.warning(f"The following sensors do not exist in the data: {', '.join(missing_sensors)}")
        new_sensor_ids = [sensor_id for sensor_id in sensor_ids if sensor_id in set(self._raw["sensor_id"])]
        self._sensor_filter = new_sensor_ids
        self.data = self._raw[self._raw["sensor_id"].isin(new_sensor_ids)].copy()

    def to_csv(self, filepath: str) -> None:
        logger.info(f"Saving data to {filepath}...")
        self.data.to_csv(filepath)
        logger.info("Saving Complete")