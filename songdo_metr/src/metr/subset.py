import os
from .components.metr_imc import TrafficData
from .components.metr_ids import IdList
from .components.graph_sensor_locations import SensorLocations
from .components.distance_imc import DistancesImc
from .components.adj_mx import AdjacencyMatrix


class MetrSubset:
    def __init__(
        self,
        raw_dir: str,
        metr_imc_filename: str = "metr-imc.h5",
        metr_ids_filename: str = "metr_ids.txt",
        graph_sensor_loc_filename: str = "graph_sensor_locations.csv",
        distances_imc_filename: str = "distances_imc_2024.csv",
        adj_mx_filename: str = "adj_mx.pkl",
    ) -> None:
        self.metr_imc_path = os.path.join(raw_dir, metr_imc_filename)
        self.metr_ids_path = os.path.join(raw_dir, metr_ids_filename)
        self.graph_sensor_loc_path = os.path.join(raw_dir, graph_sensor_loc_filename)
        self.distances_imc_path = os.path.join(raw_dir, distances_imc_filename)
        self.adj_mx_path = os.path.join(raw_dir, adj_mx_filename)
        self.metr_imc_filename = metr_imc_filename
        self.metr_ids_filename = metr_ids_filename
        self.graph_sensor_loc_filename = graph_sensor_loc_filename
        self.distances_imc_filename = distances_imc_filename
        self.adj_mx_filename = adj_mx_filename

        self.metr_imc = TrafficData.import_from_hdf(self.metr_imc_path)
        self.metr_ids = IdList.import_from_txt(self.metr_ids_path)
        self.graph_sensor_locations = SensorLocations.import_from_csv(
            self.graph_sensor_loc_path
        )
        self.distances_imc = DistancesImc.import_from_csv(self.distances_imc_path)
        self.adj_mx = AdjacencyMatrix.import_from_pickle(self.adj_mx_path)

        self.__sensor_filter = self.metr_imc.sensor_filter

    @property
    def sensor_filter(self) -> list[str]:
        return self.__sensor_filter

    @sensor_filter.setter
    def sensor_filter(self, sensor_ids: list[str], normalized_k=0.1) -> None:
        self.metr_imc.sensor_filter = sensor_ids
        self.metr_ids = IdList(self.metr_imc.sensor_filter)
        self.distances_imc.sensor_filter = sensor_ids
        self.graph_sensor_locations.sensor_filter = sensor_ids
        self.adj_mx = AdjacencyMatrix.import_from_components(
            self.metr_ids, self.distances_imc, normalized_k=normalized_k
        )

    def export(
        self,
        output_dir: str,
        missings_filename: str = "metr-imc-missings.h5",
    ) -> None:
        metr_imc_path = os.path.join(output_dir, self.metr_imc_filename)
        metr_ids_path = os.path.join(output_dir, self.metr_ids_filename)
        graph_sensor_loc_path = os.path.join(output_dir, self.graph_sensor_loc_filename)
        distances_imc_path = os.path.join(output_dir, self.distances_imc_filename)
        adj_mx_path = os.path.join(output_dir, self.adj_mx_filename)
        missings_path = os.path.join(output_dir, missings_filename)

        self.metr_imc.to_hdf(metr_imc_path, key="data")
        self.metr_ids.to_txt(metr_ids_path)
        self.graph_sensor_locations.to_csv(graph_sensor_loc_path)
        self.distances_imc.to_csv(distances_imc_path)
        self.adj_mx.to_pickle(adj_mx_path)
        self.metr_imc.is_missing_values.to_hdf(missings_path, key="data")
