import logging
import os
from typing import Optional
import zipfile
from urllib.parse import unquote

import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)


def download_file(
    url: str,
    dir_path: str,
    chunk_size: int = 1024,
    overwrite: bool = False,
) -> str:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)

    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size: Optional[int] = int(response.headers.get("content-length", -1))
    total_size = total_size if total_size != -1 else None

    # 기본 파일 이름 설정
    file_name = "data.zip"
    cd = response.headers.get("content-disposition")
    if cd and "filename=" in cd:
        file_name = unquote(cd.split("filename=")[1].split(";")[0].strip('"'))

    file_path = os.path.join(dir_path, file_name)
    logger.info(f"Downloading to {file_path}...")

    if os.path.exists(file_path):
        if overwrite:
            logger.info(f"{file_path} already exists, but will be overwritten")
        else:
            logger.info(f"{file_path} already exists, skipping download")
            return file_path

    with open(file_path, "wb") as file, tqdm(
        desc=file_name,
        total=total_size,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=chunk_size):
            file.write(data)
            bar.update(len(data))

    response.close()
    return file_path


def extract_zip_file(file_path: str, dir_path: str) -> str:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)

    target_dir = ""
    with zipfile.ZipFile(file_path, "r") as zip_ref:
        dir_names = set([filename.split("/")[0] for filename in zip_ref.namelist()])
        if len(dir_names) == 1:
            target_dir = os.path.join(dir_path, list(dir_names)[0])
            zip_ref.extractall(dir_path)
        else:
            target_dir = os.path.join(
                dir_path, os.path.splitext(os.path.basename(file_path))[0]
            )
            os.makedirs(target_dir, exist_ok=True)
            zip_ref.extractall(target_dir)

    return target_dir
