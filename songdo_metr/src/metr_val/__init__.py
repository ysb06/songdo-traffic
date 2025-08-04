"""
METR Validation Package

This package contains model validation and testing utilities for 
traffic prediction models using the METR dataset format.
"""

from .models import BasicRNN
from .test_performance import test_rnn_performance
from .utils import PathConfig

__all__ = ['BasicRNN', 'test_rnn_performance']

PATH_CONF = PathConfig("config.yaml")