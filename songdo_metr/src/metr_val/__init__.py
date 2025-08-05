"""
METR Validation Package

This package contains model validation and testing utilities for
traffic prediction models using the METR dataset format.
"""

from .utils import PathConfig


PATH_CONF = PathConfig("config.yaml")

__all__ = ["PATH_CONF"]
