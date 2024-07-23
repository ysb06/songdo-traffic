from typing import Any, Callable, List, Optional, Union
from qgis.PyQt.QtCore import QSettings, QTranslator, QCoreApplication, QStringListModel
from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtWidgets import QAction, QMessageBox
from qgis.gui import QgisInterface
from qgis.core import QgsMapLayer, QgsVectorLayer, QgsProject

from .resources import *
from .songdo_qgis_dialog import SongdoQGISDialog
import os.path
import networkx as nx

PLUGIN_GROUP_NAME = "SongdoQGIS"
NODE_KEY = "node_layer"
LINK_KEY = "link_layer"
PATH_KEY = "path_layer"


class SQAction:
    def __init__(
        self,
        iface: QgisInterface,
        action_name: str,
        icon_path=":/plugins/songdo_qgis/icon.png",
    ):
        self.iface = iface
        self.icon_path = icon_path
        self.action_name = action_name

    def __call__(self) -> None:
        raise NotImplementedError()

    @property
    def action_params(self):
        return {
            "icon_path": self.icon_path,
            "text": self.action_name,
            "callback": self,
            "parent": self.iface.mainWindow(),
        }


class SQNodeConnectorAction(SQAction):
    def __init__(
        self,
        iface: QgisInterface,
        translator: Optional[Callable[[str], str]] = None,
        name: Optional[str] = "&Connect Nodes",
    ):
        super().__init__(iface, translator(name) if translator is not None else name)

    def __call__(self) -> None:
        project = QgsProject.instance()
        # print(
        #     "Settings Loaded:",
        #     project.readEntry(PLUGIN_GROUP_NAME, "node_layer"),
        #     project.readEntry(PLUGIN_GROUP_NAME, "link_layer"),
        #     project.readEntry(PLUGIN_GROUP_NAME, "path_layer"),
        # )
        current_layer = self.iface.activeLayer()
        target_layer = project.mapLayer(project.readEntry(PLUGIN_GROUP_NAME, "node_layer")[0])
        if current_layer == target_layer:
            print("Same layer selected")
        else:
            print("Different layer selected")
        # Todo: Path 레이어로부터 그래프를 만들고 선택된 두 노드 사이의 최단 경로를 찾아서 링크 레이어에 링크를 추가


class SQSettingAction(SQAction):
    def __init__(
        self,
        iface: QgisInterface,
        translator: Optional[Callable[[str], str]] = None,
        name: Optional[str] = "&Settings...",
    ):
        super().__init__(iface, translator(name) if translator is not None else name)

    def __call__(self) -> None:
        project = QgsProject.instance()
        setting_window = SongdoQGISDialog(
            group_name=PLUGIN_GROUP_NAME,
            node_key=NODE_KEY,
            link_key=LINK_KEY,
            path_key=PATH_KEY,
        )
        setting_window.show()
        result = setting_window.exec()
        if result:
            project.writeEntry(
                PLUGIN_GROUP_NAME,
                NODE_KEY,
                setting_window.nodeSelector.currentData(),
            )
            project.writeEntry(
                PLUGIN_GROUP_NAME,
                LINK_KEY,
                setting_window.outputSelector.currentData(),
            )
            project.writeEntry(
                PLUGIN_GROUP_NAME,
                PATH_KEY,
                setting_window.pathSelector.currentData(),
            )
