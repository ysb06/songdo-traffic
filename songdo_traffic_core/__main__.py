import pandas as pd
import networkx as nx
import pickle
import numpy as np

from .visualization.metr.gis import SensorLink

SensorLink(
    "./datasets/METRLA/distances_la_2012.csv",
    "./datasets/METRLA/graph_sensor_locations.csv",
).to_file("./datasets/METRLA/miscellaneous")
