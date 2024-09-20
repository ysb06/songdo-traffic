from typing import List, Optional
import pandas as pd
import logging
import os

logger = logging.getLogger(__name__)


class MetrIds:
    @staticmethod
    def import_from_traffic_data(traffic_data: pd.DataFrame) -> "MetrIds":
        ids = traffic_data.columns.tolist()
        return MetrIds(ids)

    def __init__(self, raw: Optional[List[str]] = None) -> None:
        self.__raw: List[str] = raw if raw else []

    @property
    def data_exists(self) -> bool:
        return len(self.__raw) > 0

    def to_txt(self, dir_path: str, filename: str = "metr_ids.txt") -> None:
        logger.info(f"Saving METR-IMC IDs to {os.path.join(dir_path, filename)}...")
        ids_str = ",".join(str(id) for id in self.__raw)
        with open(os.path.join(dir_path, filename), "w") as file:
            file.write(ids_str)
        logger.info("Complete")

    @property
    def id_list(self) -> List[str]:
        return self.__raw
