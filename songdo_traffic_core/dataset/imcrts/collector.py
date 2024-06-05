import logging
import os
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from json.decoder import JSONDecodeError

import pandas as pd
import requests

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
    """Data.go.kr로부터 인천 도로 교통량 데이터를 특정 날짜 기간만큼 추출하고 저장"""

    def __init__(
        self,
        key: str,
        start_date: str = "20230101",
        end_date: str = "20231231",
        output_dir: str = "./datasets/imcrts/",
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
        self.output_dir: str = output_dir
        if not os.path.exists(self.output_dir):
            raise FileNotFoundError(f"Output Directory Not Found at {self.output_dir}")
        self.output_file_path = (
            os.path.join(self.output_dir, "imcrts_data.pkl"),
            os.path.join(self.output_dir, "imcrts_data.xlsx"),
        )

    def collect(self, ignore_empty: bool = False) -> None:
        """
        데이터를 수집하고 Pandas DataFrame형태로 변환 후 Pickle 및 Excel형태로 저장
        """
        data_list = []
        current_date: datetime = self.start_date

        logger.info(f"Collecting IMCRTS Data from {self.start_date} to {self.end_date}")

        day_count = 0
        while current_date <= self.end_date:
            current_date_string = current_date.strftime("%Y%m%d")
            self.params["YMD"] = current_date_string

            if day_count % 20 >= -1:
                logger.info(f"Requesting data at {current_date_string}...")

            time.sleep(0.05)
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
                logger.error(f"Failed to Getting Data at [{current_date_string}]")
                logger.error(f"Collecting Data Aborted!")
                break

            current_date += timedelta(days=1)

        df = pd.DataFrame(data_list)
        logger.info(f"Total Row Count: {len(df)}")
        logger.info("Creating Pickle...")
        df.to_pickle(os.path.join(self.output_dir, "imcrts_data.pkl"))
        logger.info("Creating Excel...")
        df.to_excel(os.path.join(self.output_dir, "imcrts_data.xlsx"))

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

            if len(raw["response"]["body"]["items"]) > 0:
                data = raw["response"]["body"]["items"]

                if len(data) > MAX_ROW_COUNT:
                    message = f"Length of Data at {params['YMD']} is {data['response']['body']['items']} but sliced to {MAX_ROW_COUNT}"
                    logger.warning(message)
            else:
                logger.warning(f"No data at {params['YMD']}")
        else:
            print(res.text)

        return (res.status_code, data)
