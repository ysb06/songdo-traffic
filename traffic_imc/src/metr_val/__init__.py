"""
METR Validation Package

This package contains model validation and testing utilities for
traffic prediction models using the METR dataset format.
"""

from .utils import PathConfig
from . import mlcaformer
from . import stgcn
# from . import tassgn  # Disabled: requires metr.datasets.tassgn


PATH_CONF = PathConfig("../config.yaml")

__all__ = ["PATH_CONF", "mlcaformer", "stgcn"]
