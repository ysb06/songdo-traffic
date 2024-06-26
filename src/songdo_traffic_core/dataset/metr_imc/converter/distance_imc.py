from typing import List
import geopandas as gpd
import pandas as pd
from tqdm import tqdm
import networkx as nx
import logging
from shapely.geometry import LineString

logger = logging.getLogger(__name__)


class DistancesImc:
    """distances_imc_2024.csv"""

    def __init__(
        self,
        road_data: gpd.GeoDataFrame,  # 표준노드링크 링크 데이터
        turn_info: gpd.GeoDataFrame,  # 회전 제한 정보
        target_ids: List[str],  # 교통량 데이터에 있는 id
        distance_limit: float = 3000,  # m 단위
    ) -> None:
        self.road_data = road_data.to_crs(epsg=5186)
        self.target_ids = target_ids
        self.turn_info = turn_info
        self.distance_limit = distance_limit
        self.G = self.__generate_graph(self.road_data)
        logger.info(f"Nodes: {len(self.G.nodes)}, Edges: {len(self.G.edges)}")
        self.G = self.__apply_turn_restrictions(self.G, self.turn_info)
        logger.info(f"Nodes: {len(self.G.nodes)}, Edges: {len(self.G.edges)}")
        self.distances = self.__generate_distance_data(
            self.G, self.target_ids, self.distance_limit
        )

    def __generate_distance_data(
        self, G: nx.DiGraph, target_ids: List[str], distance_limit: float
    ):
        result = []
        for from_id in tqdm(target_ids, desc="Generating Distances"):
            lengths = nx.single_source_dijkstra_path_length(
                G, source=from_id, cutoff=distance_limit, weight="length"
            )
            for to_id, distance in lengths.items():
                if to_id in target_ids:
                    result.append({"from": from_id, "to": to_id, "distance": distance})
        return pd.DataFrame(result)

    def __apply_turn_restrictions(
        self, G: nx.DiGraph, turn_info: gpd.GeoDataFrame
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

    def __generate_graph(self, road_data: gpd.GeoDataFrame):
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

    def to_csv(self, dir_path: str) -> None:
        logger.info(f"Saving distances data to {dir_path}/distances_imc_2024.csv...")
        self.distances.to_csv(f"{dir_path}/distances_imc_2024.csv", index=False)
        logger.info("Complete")
