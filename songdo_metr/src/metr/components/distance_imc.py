import os
from typing import List
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class DistancesImc:
    @staticmethod
    def import_from_csv(filepath: str) -> "DistancesImc":
        raw = pd.read_csv(filepath, dtype={"from": str, "to": str})
        return DistancesImc(raw)
    
    def __init__(self, raw: pd.DataFrame) -> None:
        self._raw = raw
        # Initialize sensor_filter as a set of unique sensor IDs from 'from' and 'to' columns
        self._sensor_filter = self._raw[["from", "to"]].stack().unique()
        self.data = self._raw.copy()

    @property
    def sensor_filter(self) -> List[str]:
        return list(self._sensor_filter)

    @sensor_filter.setter
    def sensor_filter(self, sensor_ids: List[str]) -> None:
        target_id_set = set(sensor_ids)
        original_id_set = set(self._raw[["from", "to"]].stack().unique())
    
        missing_sensors = target_id_set - original_id_set
        if missing_sensors:
            logger.warning(f"The following sensors do not exist in the data: {', '.join(missing_sensors)}")
        new_sensor_ids = target_id_set & original_id_set

        self._sensor_filter = new_sensor_ids
        # Filter data where both 'from' and 'to' are in new_sensor_ids
        self.data = self._raw[
            self._raw["from"].isin(new_sensor_ids) & self._raw["to"].isin(new_sensor_ids)
        ].copy()

    def to_csv(self, dir_path: str, filename: str = "distance_imc_2024.csv") -> None:
        filepath = os.path.join(dir_path, filename)
        logger.info(f"Saving data to {filepath}...")
        self.data.to_csv(filepath, index=False)
        logger.info("Saving Complete")