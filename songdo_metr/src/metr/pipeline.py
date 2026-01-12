import logging
import os
from typing import Optional

import geopandas as gpd
import pandas as pd

from metr.components import (
    AdjacencyMatrix,
    DistancesImc,
    IdList,
    Metadata,
    SensorLocations,
    TrafficData,
)
from metr.components.metr_imc.outlier import (
    RemovingWeirdZeroOutlierProcessor,
    TrafficCapacityAbsoluteOutlierProcessor,
)
from metr.imcrts.collector import IMCRTSCollector
from metr.nodelink.converter import NodeLink
from metr.nodelink.downloader import download_nodelink
from metr.utils import PathConfig

logger = logging.getLogger(__name__)
PATH_CONF = PathConfig.from_yaml("../config.yaml")
PATH_CONF.create_directories()
PATH_SUBSET_CONF = PathConfig.from_yaml("../config_filtered.yaml")


# Other Settings
PDP_KEY = os.environ.get("PDP_KEY")
NODELINK_RAW_URL = (
    "https://www.its.go.kr/opendata/nodelinkFileSDownload/DF_180/0"  # 2022-12-28
)
TARGET_REGION_CODES = [
    "161",
    "162",
    "163",
    "164",
    "165",
    "166",
    "167",
    "168",
    "169",
]  # All Incheon Regions
IMCRTS_START_DATE = "20230101"
IMCRTS_END_DATE = "20251231"


# def generate_subset_dataset(
#     target_nodelink: list[str],
#     save_dir_path: str,
# ):
#     metr_imc_filename = PATH_CONF.raw["dataset"]["filenames"]["metr_imc"]
#     sensor_ids_filename = PATH_CONF.raw["dataset"]["filenames"]["sensor_ids"]
#     metadata_filename = PATH_CONF.raw["dataset"]["filenames"]["metadata"]
#     sensor_locations_filename = PATH_CONF.raw["dataset"]["filenames"][
#         "sensor_locations"
#     ]
#     distances_filename = PATH_CONF.raw["dataset"]["filenames"]["distances"]
#     adjacency_matrix_filename = PATH_CONF.raw["dataset"]["filenames"][
#         "adjacency_matrix"
#     ]

#     metr_imc_save_path = os.path.join(save_dir_path, metr_imc_filename)
#     sensor_ids_save_path = os.path.join(save_dir_path, sensor_ids_filename)
#     metadata_save_path = os.path.join(save_dir_path, metadata_filename)
#     sensor_locations_save_path = os.path.join(save_dir_path, sensor_locations_filename)
#     distances_save_path = os.path.join(save_dir_path, distances_filename)
#     adj_mx_save_path = os.path.join(save_dir_path, adjacency_matrix_filename)

#     traffic_data = TrafficData.import_from_hdf(PATH_CONF.metr_imc_path)
#     wz_outlier_processor = RemovingWeirdZeroOutlierProcessor()
#     traffic_data.data = wz_outlier_processor.process(traffic_data.data)
#     traffic_data.select_sensors(target_nodelink)
#     traffic_data.to_hdf(metr_imc_save_path)

#     generate_dataset(
#         traffic_data_path=metr_imc_save_path,
#         ids_output_path=sensor_ids_save_path,
#         metadata_output_path=metadata_save_path,
#         sensor_locations_output_path=sensor_locations_save_path,
#         distances_output_path=distances_save_path,
#         adj_mx_output_path=adj_mx_save_path,
#     )


def generate_raw_dataset():
    # Generating Core Files
    # generate_nodelink_raw()
    # generate_imcrts_raw()
    generate_metr_imc_raw()
    generate_dataset()

    # Generating Misc
    generate_metr_imc_shapefile()
    generate_distances_shapefile()

    # Generating excel files
    generate_metr_imc_excel()


def generate_subset_dataset(
    subset_path_conf: PathConfig = PATH_SUBSET_CONF,
    target_nodelinks_path: Optional[str] = None,
    target_data_start: Optional[str] = None,
    target_data_end: Optional[str] = None,
):
    """
    기존 raw 데이터셋에서 공간/시간 필터링하여 subset 데이터셋을 생성합니다.

    Args:
        subset_path_conf: subset 데이터셋 경로를 정의한 PathConfig 객체
        target_nodelinks_path: 필터링할 도로 shapefile 경로 (None이면 전체 사용)
        target_data_start: 시작 날짜 (None이면 전체 사용)
        target_data_end: 종료 날짜 (None이면 전체 사용)
    """
    # 1. 디렉토리 생성
    subset_path_conf.create_directories()
    logger.info(f"Generating subset dataset at: {subset_path_conf.root_dir_path}")

    # 2. 전체 raw 데이터 로드
    logger.info("Loading raw METR-IMC data...")
    traffic_data = TrafficData.import_from_hdf(PATH_CONF.metr_imc_path)
    df = traffic_data.data
    logger.info(f"Original data: {len(df)} rows, {len(df.columns)} sensors")

    # 3. 공간 필터링 (shapefile에서 LINK_ID 추출 후 직접 열 선택)
    if target_nodelinks_path:
        logger.info(f"Filtering sensors from shapefile: {target_nodelinks_path}")
        target_roads = gpd.read_file(target_nodelinks_path)
        target_link_ids = target_roads["LINK_ID"].tolist()
        # DataFrame에서 교집합 열만 선택 (존재하지 않는 LINK_ID는 무시)
        valid_link_ids = [lid for lid in target_link_ids if lid in df.columns]
        df = df[valid_link_ids]
        logger.info(f"After spatial filtering: {len(df.columns)} sensors")

    # 4. 시간 필터링 (DatetimeIndex 기반)
    if target_data_start or target_data_end:
        logger.info(f"Filtering time range: {target_data_start} ~ {target_data_end}")
        df = df.loc[target_data_start:target_data_end]
        logger.info(f"After temporal filtering: {len(df)} rows")

    # 5. 필터링된 데이터 저장
    traffic_data.data = df
    logger.info(f"Saving filtered traffic data to {subset_path_conf.metr_imc_path}")
    traffic_data.to_hdf(subset_path_conf.metr_imc_path)

    # 6. generate_dataset() 호출 (subset PathConfig의 경로 사용)
    logger.info("Generating dataset components...")
    generate_dataset(
        traffic_data_path=subset_path_conf.metr_imc_path,
        nodelink_link_path=PATH_CONF.nodelink_link_path,  # raw 것 사용
        nodelink_turn_path=PATH_CONF.nodelink_turn_path,  # raw 것 사용
        ids_output_path=subset_path_conf.sensor_ids_path,
        metadata_output_path=subset_path_conf.metadata_path,
        sensor_locations_output_path=subset_path_conf.sensor_locations_path,
        distances_output_path=subset_path_conf.distances_path,
        adj_mx_output_path=subset_path_conf.adj_mx_path,
    )

    # 7. shapefile 생성
    logger.info("Generating shapefiles...")
    generate_metr_imc_shapefile(
        metr_imc_path=subset_path_conf.metr_imc_path,
        node_link_path=PATH_CONF.nodelink_link_path,
        output_path=subset_path_conf.metr_shapefile_path,
    )

    generate_distances_shapefile(
        distances_path=subset_path_conf.distances_path,
        sensor_locations_path=subset_path_conf.sensor_locations_path,
        output_path=subset_path_conf.distances_shapefile_path,
    )

    logger.info(f"Subset dataset generation completed: {subset_path_conf.root_dir_path}")


def generate_metr_imc_excel(
    metr_imc_path: str = PATH_CONF.metr_imc_path,
    output_dir: str = PATH_CONF.misc_dir_path,
    max_rows_per_file: int = 1000000,
):
    """
    metr_imc.h5 데이터를 엑셀 파일로 저장합니다.
    데이터가 엑셀 행 제한(1,048,576)을 초과하면 여러 파일로 분할합니다.

    Args:
        metr_imc_path: HDF5 파일 경로
        output_dir: 출력 디렉토리 (None이면 HDF5 파일과 같은 디렉토리)
        max_rows_per_file: 파일당 최대 행 수 (기본값: 1,000,000)
    """
    logger.info("Loading METR-IMC data from HDF5...")
    traffic_data = TrafficData.import_from_hdf(metr_imc_path)
    df = traffic_data.data

    # 출력 디렉토리 설정
    if output_dir is None:
        output_dir = os.path.dirname(metr_imc_path)

    total_rows = len(df)
    logger.info(f"Total rows: {total_rows}, Total sensors: {len(df.columns)}")

    # 데이터가 엑셀 제한보다 작으면 단일 파일로 저장
    if total_rows <= max_rows_per_file:
        output_path = os.path.join(output_dir, "metr-imc.xlsx")
        logger.info(f"Saving to {output_path}...")
        df.to_excel(output_path, engine="openpyxl")
        logger.info("Excel file saved successfully")
    else:
        # 여러 파일로 분할
        num_files = (total_rows + max_rows_per_file - 1) // max_rows_per_file
        logger.info(f"Data exceeds Excel limit. Splitting into {num_files} files...")

        for i in range(num_files):
            start_idx = i * max_rows_per_file
            end_idx = min((i + 1) * max_rows_per_file, total_rows)
            df_chunk = df.iloc[start_idx:end_idx]

            output_path = os.path.join(output_dir, f"metr-imc_part{i+1:02d}.xlsx")
            logger.info(
                f"Saving part {i+1}/{num_files} ({end_idx - start_idx} rows) to {output_path}..."
            )
            df_chunk.to_excel(output_path, engine="openpyxl")

        logger.info(f"All {num_files} Excel files saved successfully")


def generate_distances_shapefile(
    distances_path: str = PATH_CONF.distances_path,
    sensor_locations_path: str = PATH_CONF.sensor_locations_path,
    output_path: str = PATH_CONF.distances_shapefile_path,
):
    distances = DistancesImc.import_from_csv(distances_path)
    sensor_locations = SensorLocations.import_from_csv(sensor_locations_path)
    distances.to_shapefile(sensor_locations.data, filepath=output_path)


def generate_dataset(
    traffic_data_path: str = PATH_CONF.metr_imc_path,
    nodelink_link_path: str = PATH_CONF.nodelink_link_path,
    nodelink_turn_path: str = PATH_CONF.nodelink_turn_path,
    ids_output_path: str = PATH_CONF.sensor_ids_path,
    metadata_output_path: str = PATH_CONF.metadata_path,
    sensor_locations_output_path: str = PATH_CONF.sensor_locations_path,
    distances_output_path: str = PATH_CONF.distances_path,
    adj_mx_output_path: str = PATH_CONF.adj_mx_path,
):
    traffic_data = TrafficData.import_from_hdf(traffic_data_path)

    # Sensor IDs
    metr_ids = IdList(traffic_data.data.columns.to_list())
    metr_ids.to_txt(ids_output_path)

    # Metadata
    metadata = Metadata.import_from_nodelink(nodelink_link_path)
    metadata.sensor_filter = metr_ids.data  # 수정 필요
    metadata.to_hdf(metadata_output_path)

    # Sensor Locations
    sensor_locations = SensorLocations.import_from_nodelink(nodelink_link_path)
    sensor_locations.sensor_filter = metr_ids.data
    sensor_locations.to_csv(sensor_locations_output_path)

    # Distances
    distances = DistancesImc.import_from_nodelink(
        nodelink_link_path,
        nodelink_turn_path,
        target_ids=metr_ids.data,
        distance_limit=9000,
    )
    distances.to_csv(distances_output_path)

    # Adjacency Matrix
    adj_mx: AdjacencyMatrix = AdjacencyMatrix.import_from_components(
        metr_ids, distances
    )
    adj_mx.to_pickle(adj_mx_output_path)


def generate_nodelink_raw(
    nodelink_url: str = NODELINK_RAW_URL,
    region_codes: list[str] = TARGET_REGION_CODES,
    download_target_dir: str = PATH_CONF.nodelink_dir_path,
    node_output_path: str = PATH_CONF.nodelink_node_path,
    link_output_path: str = PATH_CONF.nodelink_link_path,
    turn_output_path: str = PATH_CONF.nodelink_turn_path,
):
    logger.info("Downloading Node-Link Data...")
    nodelink_raw_path = download_nodelink(download_target_dir, nodelink_url)
    nodelink_data = NodeLink(nodelink_raw_path).filter_by_gu_codes(region_codes)
    nodelink_data.export(
        node_output_path=node_output_path,
        link_output_path=link_output_path,
        turn_output_path=turn_output_path,
    )
    logger.info("Downloading Done")


def generate_imcrts_raw(
    api_key: Optional[str] = PDP_KEY,
    start_date: str = IMCRTS_START_DATE,
    end_date: str = IMCRTS_END_DATE,
    imcrts_output_path: str = PATH_CONF.imcrts_path,
):
    logger.info("Collecting IMCRTS Data...")
    if api_key is None:
        raise ValueError("PDP_KEY 환경 변수가 설정되어 있지 않습니다.")
    
    collector = IMCRTSCollector(
        api_key=api_key,
        start_date=start_date,
        end_date=end_date,
    )
    collector.collect(ignore_empty=True)
    collector.to_pickle(imcrts_output_path)
    logger.info("Collecting Done")


def generate_metr_imc_raw(
    road_data_path: str = PATH_CONF.nodelink_link_path,
    traffic_data_path: str = PATH_CONF.imcrts_path,
    metr_imc_path: str = PATH_CONF.metr_imc_path,
):
    road_data: gpd.GeoDataFrame = gpd.read_file(road_data_path)
    traffic_data = TrafficData.import_from_pickle(traffic_data_path)

    logger.info("Matching Link IDs...")
    traffic_data.select_sensors(road_data["LINK_ID"].tolist())
    traffic_data.to_hdf(metr_imc_path)
    logger.info("Matching Done")


def generate_metr_imc_shapefile(
    metr_imc_path: str = PATH_CONF.metr_imc_path,
    node_link_path: str = PATH_CONF.nodelink_link_path,
    output_path: str = PATH_CONF.metr_shapefile_path,
):
    traffic_data = TrafficData.import_from_hdf(metr_imc_path)
    road_data: gpd.GeoDataFrame = gpd.read_file(node_link_path)
    traffic_link_ids = set(traffic_data.data.columns)
    filtered_roads = road_data[road_data["LINK_ID"].isin(traffic_link_ids)].copy()
    filtered_roads.to_file(output_path)
