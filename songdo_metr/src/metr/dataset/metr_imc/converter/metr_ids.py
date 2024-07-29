from typing import List, Optional
import pandas as pd
import logging
import os

logger = logging.getLogger(__name__)


class MetrIds:
    def __init__(self, raw: Optional[List[str]] = None) -> None:
        self.ids: List[str] = raw if raw else []
    
    @property
    def data_exists(self) -> bool:
        return len(self.ids) > 0

    def import_from_traffic_data(self, traffic_data: pd.DataFrame) -> None:
        self.ids = traffic_data.columns.tolist()

    def to_txt(self, dir_path: str, filename: str = "metr_ids.txt") -> None:
        logger.info(f"Saving METR-IMC IDs to {os.path.join(dir_path, filename)}...")
        ids_str = ",".join(str(id) for id in self.ids)
        with open(os.path.join(dir_path, filename), "w") as file:
            file.write(ids_str)
        logger.info("Complete")

    @property
    def id_list(self) -> List[str]:
        return self.ids
