import logging
import os
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from json.decoder import JSONDecodeError

import pandas as pd
import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)


def load_key(path: str) -> str:
    try:
        with open(path, "r") as f:
            return f.read().strip()
    except FileNotFoundError:
        logger.error(f"Key File Not Found at {path}")
        return ""


SERVICE_URL = "http://apis.data.go.kr/6280000/ICRoadVolStat/NodeLink_Trfc_DD"
MAX_ROW_COUNT = 5000


class IMCRTSExcelConverter:
    """IMCRTS Collector에서 Pickle로 저장된 데이터를 Excel로 변환, openpyxl을 설치하지 못해 에러가 발생했을 때 사용"""

    def __init__(
        self, output_dir: str = "./datasets/imcrts/", filename: str = "imcrts_data.pkl"
    ) -> None:
        self.output_dir = output_dir
        self.filename = filename
        self.filepath = os.path.join(self.output_dir, self.filename)
        logger.info(f"Loading Data from {self.filepath}...")
        self.data: pd.DataFrame = pd.read_pickle(self.filepath)

    def export(self):
        logger.info("Exporting Data to Excel...")
        self.data.to_excel(os.path.join(self.output_dir, "imcrts_data.xlsx"))


class IMCRTSCollector:
    def __init__(
        self,
        key: str,
        start_date: str = "20230101",
        end_date: str = "20231231", # Include the end date
    ) -> None:
        logger.info("Collecting...")
        self.params = {
            "serviceKey": key,
            "pageNo": 1,
            "numOfRows": MAX_ROW_COUNT,
            "YMD": "20240101",
        }
        self.start_date: datetime = datetime.strptime(start_date, "%Y%m%d")
        self.end_date: datetime = datetime.strptime(end_date, "%Y%m%d")

    def __check_overwrite(self, output_dir: str, file_name: str, total: int) -> bool:
        pickle_path = os.path.join(output_dir, file_name + ".pkl")
        excel_path = os.path.join(output_dir, file_name + ".xlsx")

        if os.path.exists(pickle_path):
            logger.warning(f"Pickle File Already Exists at {pickle_path}")
            logger.info(f"Checking Pickle File...")
            temp = pd.read_pickle(pickle_path)
            logger.info(f"{len(temp)} Rows in Pickle File")
            if len(temp) == total:
                logger.info("Pickle File is Valid")
            else:
                logger.warning("Pickle File is Invalid")
                return False
        else:
            return False

        if os.path.exists(excel_path):
            logger.warning(f"Excel File Already Exists at {excel_path}")
            logger.info(f"Checking Excel File...")
            temp = pd.read_excel(excel_path)
            logger.info(f"{len(temp)} Rows in Excel File")
            if len(temp) == total:
                logger.info("Excel File is Valid")
            else:
                logger.warning("Excel File is Invalid")
                return False
        else:
            return False

        return True

    def collect(
        self,
        output_dir: str,
        ignore_empty: bool = False,
        req_delay: float = 0.05,
        file_name: str = "imcrts_data",
        overwrite: bool = False,
    ) -> None:
        """
        데이터를 수집하고 Pandas DataFrame형태로 변환 후 Pickle 및 Excel형태로 저장
        """
        if not os.path.exists(output_dir):
            logger.info(f"Creating Directory at {output_dir}")
            os.makedirs(output_dir, exist_ok=True)

        total_size = (self.end_date - self.start_date).days + 1
        if not overwrite and self.__check_overwrite(output_dir, file_name, total_size):
            logger.error("Skipping collecting due to existing valid files")
            return

        data_list = []
        current_date: datetime = self.start_date

        logger.info(f"Collecting IMCRTS Data from {self.start_date} to {self.end_date}")
        with tqdm(total=total_size) as bar:
            while current_date <= self.end_date:
                current_date_string = current_date.strftime("%Y%m%d")
                self.params["YMD"] = current_date_string
                bar.set_description(current_date_string, refresh=True)

                time.sleep(req_delay)
                code, data = self.get_data(self.params)
                if code == 200:
                    if data is not None:
                        data_list.extend(data)
                    else:
                        if ignore_empty:
                            logger.warning("Skipping...")
                        else:
                            logger.error("Aborted due to empty data")
                            break
                else:
                    logger.error(f"Code: {code}")
                    logger.error(f"Failed to Get Data at [{current_date_string}]")
                    logger.error(f"Collecting Data Aborted!")
                    break

                current_date += timedelta(days=1)
                bar.update(1)

        df = pd.DataFrame(data_list)
        logger.info(f"Total Row Count: {len(df)}")
        logger.info("Creating Pickle...")
        df.to_pickle(os.path.join(output_dir, file_name + ".pkl"))
        logger.info("Creating Excel...")
        df.to_excel(os.path.join(output_dir, file_name + ".xlsx"))

    def get_data(
        self, params: Dict[str, Any]
    ) -> Tuple[int, Optional[List[Dict[str, Any]]]]:
        """Request Data from Data Server
        SERVICE_URL로부터 GET 데이터 요청을 수행한다.

        Args:
            params (Dict[str, Any]): Parameters for Request

        Returns:
            Tuple[int, Optional[List[Dict[str, Any]]]]: Result of Data Request
        """
        res = requests.get(url=SERVICE_URL, params=params)
        data: Optional[List[Dict[str, Any]]] = None
        if res.status_code == 200:
            try:
                raw = res.json()
            except JSONDecodeError:
                logger.error("JSON Decoding Failed")
                if "SERVICE_KEY_IS_NOT_REGISTERED_ERROR" in res.text:
                    logger.error("You may use not valid service key")
                return 0, []

            if raw["response"]["header"]["resultCode"] != "00":
                logger.warning(
                    f"Error Code: {raw['response']['header']['resultCode']}. You need to check the reason of error."
                )

            if "items" in raw["response"]["body"] and raw["response"]["body"]["items"]:
                data = raw["response"]["body"]["items"]

                if len(data) > MAX_ROW_COUNT:
                    message = f"Length of Data at {params['YMD']} is {len(data)} but sliced to {MAX_ROW_COUNT}"
                    logger.warning(message)
            else:
                logger.warning(f"No data at {params['YMD']}")
        else:
            logger.error(f"Request failed with status code {res.status_code}")
            logger.error(res.text)

        return (res.status_code, data)
