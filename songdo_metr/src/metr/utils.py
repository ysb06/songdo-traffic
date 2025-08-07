import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Union
import yaml


@dataclass
class PathConfig:
    """Path configuration using dataclass"""

    # Root directory
    root_dir_path: str

    # Core dataset file paths
    metr_imc_path: str
    sensor_ids_path: str
    metadata_path: str
    sensor_locations_path: str
    distances_path: str
    adj_mx_path: str

    # Nodelink paths
    nodelink_dir_path: str
    nodelink_node_path: str
    nodelink_link_path: str
    nodelink_turn_path: str

    # IMCRTS paths
    imcrts_dir_path: str
    imcrts_path: str

    # Miscellaneous paths
    misc_dir_path: str
    imcrts_excel_path: str
    metr_excel_path: str
    metr_shapefile_path: str
    distances_shapefile_path: str

    # Path Raws
    raw: Dict[str, Any]

    @classmethod
    def from_yaml(cls, config_path: str = None) -> "PathConfig":
        """Create PathConfig from YAML file"""
        if config_path is None:
            # Default config path relative to this file
            current_dir = Path(__file__).parent
            config_path = current_dir.parent.parent / "config.yaml"

        with open(config_path, "r", encoding="utf-8") as f:
            config: Dict[str, Any] = yaml.safe_load(f)

        return cls._build_from_config(config)

    @classmethod
    def _build_from_config(cls, config: Dict[str, Any]) -> "PathConfig":
        """Build PathConfig from configuration dictionary"""
        root_dir = config["root_dir"]

        # Core dataset file paths
        dataset_filenames = config["dataset"]["filenames"]
        metr_imc_path = os.path.join(root_dir, dataset_filenames["metr_imc"])
        sensor_ids_path = os.path.join(root_dir, dataset_filenames["sensor_ids"])
        metadata_path = os.path.join(root_dir, dataset_filenames["metadata"])
        sensor_locations_path = os.path.join(
            root_dir, dataset_filenames["sensor_locations"]
        )
        distances_path = os.path.join(root_dir, dataset_filenames["distances"])
        adj_mx_path = os.path.join(root_dir, dataset_filenames["adjacency_matrix"])

        # Nodelink paths
        nodelink_dir = os.path.join(root_dir, config["nodelink"]["dir"])
        nodelink_filenames = config["nodelink"]["filenames"]
        nodelink_node_path = os.path.join(nodelink_dir, nodelink_filenames["node"])
        nodelink_link_path = os.path.join(nodelink_dir, nodelink_filenames["link"])
        nodelink_turn_path = os.path.join(nodelink_dir, nodelink_filenames["turn"])

        # IMCRTS paths
        imcrts_dir = os.path.join(root_dir, config["imcrts"]["dir"])
        imcrts_filenames = config["imcrts"]["filenames"]
        imcrts_path = os.path.join(imcrts_dir, imcrts_filenames["data"])

        # Miscellaneous paths
        misc_dir = os.path.join(root_dir, config["misc"]["dir"])
        misc_filenames = config["misc"]["filenames"]
        imcrts_excel_path = os.path.join(misc_dir, misc_filenames["imcrts_excel"])
        metr_excel_path = os.path.join(misc_dir, misc_filenames["metr_excel"])
        metr_shapefile_path = os.path.join(misc_dir, misc_filenames["metr_shape"])
        distances_shapefile_path = os.path.join(
            misc_dir, misc_filenames["distances_shape"]
        )

        return cls(
            root_dir_path=root_dir,
            metr_imc_path=metr_imc_path,
            sensor_ids_path=sensor_ids_path,
            metadata_path=metadata_path,
            sensor_locations_path=sensor_locations_path,
            distances_path=distances_path,
            adj_mx_path=adj_mx_path,
            nodelink_dir_path=nodelink_dir,
            nodelink_node_path=nodelink_node_path,
            nodelink_link_path=nodelink_link_path,
            nodelink_turn_path=nodelink_turn_path,
            imcrts_dir_path=imcrts_dir,
            imcrts_path=imcrts_path,
            misc_dir_path=misc_dir,
            imcrts_excel_path=imcrts_excel_path,
            metr_excel_path=metr_excel_path,
            metr_shapefile_path=metr_shapefile_path,
            distances_shapefile_path=distances_shapefile_path,
            raw=config,
        )

    def create_directories(self):
        """Create necessary directories"""
        dirs_to_create = [
            self.misc_dir_path,
            self.nodelink_dir_path,
            self.imcrts_dir_path,
        ]

        for dir_path in dirs_to_create:
            os.makedirs(dir_path, exist_ok=True)
