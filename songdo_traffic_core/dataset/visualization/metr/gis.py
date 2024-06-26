from typing import Dict, Set
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString, Point
from tqdm import tqdm


class SensorLink:
    def __init__(self, distance_csv_path: str, sensor_loc_csv_path: str) -> None:
        self.sensor_distance_df = pd.read_csv(distance_csv_path)
        self.sensor_location_df = pd.read_csv(sensor_loc_csv_path, index_col="index")

        self.node_data = self.__generate_node_data(self.sensor_location_df)

        node_dict = self.node_data.set_index("NODE_ID")["geometry"].to_dict()
        self.link_data = self.__generate_link_data(
            node_dict, set(self.node_data["NODE_ID"])
        )

    def to_file(self, output_dir: str) -> None:
        self.node_data.to_file(f"{output_dir}/sensor_nodes.shp", encoding="utf-8")
        self.link_data.to_file(f"{output_dir}/sensor_links.shp", encoding="utf-8")

    def __generate_node_data(self, sensor_loc_df: pd.DataFrame) -> gpd.GeoDataFrame:
        data = gpd.GeoDataFrame(
            sensor_loc_df["sensor_id"],
            geometry=gpd.points_from_xy(
                sensor_loc_df["longitude"], sensor_loc_df["latitude"]
            ),
            crs="EPSG:4326",
        )
        data = data.rename(columns={"sensor_id": "NODE_ID"})

        return data

    def __generate_link_data(
        self,
        node_dict: Dict[str, Point],
        node_set: Set[str],
        link_list: gpd.GeoDataFrame = None,
        delete_zero_length: bool = True,
    ) -> gpd.GeoDataFrame:
        # 굳이 매개변수 받지 말고 self 매개변수로 받아서 처리하는게 더 좋을 것 같음
        link_data = self.sensor_distance_df[
            (self.sensor_distance_df["from"].isin(node_set))
            & (self.sensor_distance_df["to"].isin(node_set))
        ]
        link_data.reset_index(drop=True, inplace=True)

        if link_list is None:
            pass  # 현재는 미구현, link_list에서 LINK_ID를 검색하여 self.link_data.index를 생성하는 방식으로 구현
            # 표준노드링크를 사용한 데이터에서 사용 예정

        data = {"LINK_ID": [], "F_NODE": [], "T_NODE": [], "COST": [], "geometry": []}
        for idx, from_node, to_node, cost in tqdm(
            link_data.itertuples(),
            total=link_data.shape[0],
        ):
            link_line = LineString([node_dict[from_node], node_dict[to_node]])
            if delete_zero_length and link_line.length <= 0:
                continue

            data["LINK_ID"].append(idx)
            data["F_NODE"].append(from_node)
            data["T_NODE"].append(to_node)
            data["COST"].append(cost)  # This may same as geometry length
            data["geometry"].append(link_line)
            
        data = gpd.GeoDataFrame(data, crs="EPSG:4326")

        return data

# METR-LA의 속도는 mile/h라고 하지만, Distance의 경우 m 단위로 보임
# 아마도 Distance의 경우 처음 데이터 이후에 추가된 것으로 보임