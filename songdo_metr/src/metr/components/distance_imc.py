import logging
import os
from typing import List

import geopandas as gpd
import networkx as nx
import pandas as pd
from shapely.geometry import LineString
from tqdm import tqdm
from shapely.geometry import LineString

logger = logging.getLogger(__name__)


class DistancesImc:
    @staticmethod
    def import_from_csv(filepath: str) -> "DistancesImc":
        raw = pd.read_csv(filepath, dtype={"from": str, "to": str})
        return DistancesImc(raw)

    @staticmethod
    def import_from_nodelink(
        nodelink_road_path: str,
        nodelink_turn_path: str,
        target_ids: List[str],
        distance_limit: float,
    ) -> "DistancesImc":
        road_data = gpd.read_file(nodelink_road_path)
        turn_info = gpd.read_file(nodelink_turn_path)
        graph = SensorGraph(road_data, turn_info)

        return DistancesImc(graph.generate_distance_data(target_ids, distance_limit))

    def __init__(self, raw: pd.DataFrame) -> None:
        self._raw = raw
        # Initialize sensor_filter as a set of unique sensor IDs from 'from' and 'to' columns
        self._sensor_filter = self._raw[["from", "to"]].stack().unique()
        self.data = self._raw.copy()

    def to_csv(self, filepath: str) -> None:
        logger.info(f"Saving data to {filepath}...")
        self.data.to_csv(filepath, index=False)
        logger.info("Saving Complete")

    def to_shapefile(
        self,
        sensor_locations: pd.DataFrame,
        filepath: str,
        crs: str = "EPSG:3857",
    ) -> None:
        logger.info(f"{filepath}에 Shapefile 생성 중...")

        # 센서 위치 정보를 빠른 조회를 위해 딕셔너리로 변환
        loc_dict = sensor_locations.set_index("sensor_id")[
            ["latitude", "longitude"]
        ].to_dict("index")

        # 각 from-to 쌍에 대한 LineString 생성
        geometries = []
        properties = []

        for _, row in tqdm(self.data.iterrows(), total=self.data.shape[0]):
            from_id = row["from"]
            to_id = row["to"]

            # 두 ID가 모두 위치 정보에 있는지 확인
            if from_id not in loc_dict or to_id not in loc_dict:
                logger.warning(
                    f"{from_id} 또는 {to_id}의 위치 정보가 없습니다. 건너뜁니다..."
                )
                continue

            # LineString 생성
            from_point = (loc_dict[from_id]["longitude"], loc_dict[from_id]["latitude"])
            to_point = (loc_dict[to_id]["longitude"], loc_dict[to_id]["latitude"])
            line = LineString([from_point, to_point])

            # 목록에 추가
            geometries.append(line)
            properties.append({"from_id": from_id, "to_id": to_id})

        # GeoDataFrame 생성
        gdf = gpd.GeoDataFrame(properties, geometry=geometries, crs="EPSG:4326")
        gdf.to_crs(crs, inplace=True)

        # Shapefile로 저장
        gdf.to_file(filepath)

        logger.info("Shapefile 생성 완료")


class SensorGraph:
    def __init__(
        self,
        road_data: gpd.GeoDataFrame,  # 표준노드링크 링크 데이터
        turn_info: pd.DataFrame,  # 회전 제한 정보
    ) -> None:
        logger.info("Generating Sensor Graph...")
        self._G = self._generate_graph(road_data)
        logger.info(f"Nodes: {len(self._G.nodes)}, Edges: {len(self._G.edges)}")
        logger.info("Applying Turn Restrictions...")
        self._G = self._apply_turn_restrictions(self._G, turn_info)
        logger.info(f"Nodes: {len(self._G.nodes)}, Edges: {len(self._G.edges)}")

    @property
    def graph(self) -> nx.DiGraph:
        return self._G

    def _apply_turn_restrictions(
        self, G: nx.DiGraph, turn_info: pd.DataFrame
    ) -> nx.DiGraph:
        for _, row in tqdm(
            turn_info.iterrows(),
            total=turn_info.shape[0],
            desc="Applying Turn Restrictions",
        ):
            start_id = row["ST_LINK"]
            end_id = row["ED_LINK"]
            if (
                row["TURN_TYPE"] in ["001", "011", "012"]  # 비보호 회전, U-Turn, P-Turn
                and G.has_node(start_id)
                and G.has_node(end_id)
            ):
                intersection_id = row["NODE_ID"]
                start_length = G.nodes[start_id]["geometry"].length
                end_length = G.nodes[end_id]["geometry"].length
                length = (start_length + end_length) / 2
                G.add_edge(
                    start_id,
                    end_id,
                    intersection_id=intersection_id,
                    length=length,
                )
            elif row["TURN_TYPE"] in [
                "002",  # 버스만회전
                "003",  # 회전금지
                "101",  # 좌회전금지
                "102",  # 직진금지
                "103",  # 우회전금지
            ]:
                if G.has_edge(start_id, end_id):
                    G.remove_edge(start_id, end_id)

        return G

    def _generate_graph(self, road_data: gpd.GeoDataFrame):
        G = nx.DiGraph()

        # Node 추가
        for _, start_road_row in tqdm(
            road_data.iterrows(), total=road_data.shape[0], desc="Adding Nodes"
        ):
            attr = {
                k: v
                for k, v in start_road_row.to_dict().items()
                if k not in ["LINK_ID", "F_NODE", "T_NODE"]
            }
            G.add_node(start_road_row["LINK_ID"], **attr)

        # Edge 미리 계산
        node_map = road_data.groupby("F_NODE")["LINK_ID"].apply(list).to_dict()
        road_data_dict = road_data.set_index("LINK_ID").to_dict("index")

        # Edge 추가
        for _, start_road_row in tqdm(
            road_data.iterrows(), total=road_data.shape[0], desc="Adding Edges"
        ):
            start_road_id = start_road_row["LINK_ID"]
            intersection_id = start_road_row["T_NODE"]
            if intersection_id in node_map:
                end_road_ids = node_map[intersection_id]
                for end_road_id in end_road_ids:
                    end_road_row = road_data_dict[end_road_id]
                    if start_road_row["F_NODE"] == end_road_row["T_NODE"]:
                        continue  # Basically, U-Turn is not allowed in a intersection

                    start_road_line: LineString = start_road_row["geometry"]
                    end_road_line: LineString = end_road_row["geometry"]
                    length = (start_road_line.length + end_road_line.length) / 2

                    G.add_edge(
                        start_road_id,
                        end_road_id,
                        intersection_id=intersection_id,
                        length=length,
                    )
        return G

    def generate_distance_data(self, target_ids: List[str], distance_limit: float):
        result = []
        for from_id in tqdm(target_ids, desc="Generating Distances"):
            lengths = nx.single_source_dijkstra_path_length(
                self.graph, source=from_id, cutoff=distance_limit, weight="length"
            )
            for to_id, distance in lengths.items():
                if to_id in target_ids:
                    result.append({"from": from_id, "to": to_id, "distance": distance})
        return pd.DataFrame(result)
