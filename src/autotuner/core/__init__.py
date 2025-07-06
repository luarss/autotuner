# ruff: noqa
"""
Core abstractions and interfaces for the AutoTuner framework.

This module contains the fundamental abstractions that define the architecture
of the AutoTuner system, including base classes for algorithms, evaluators,
and runners, as well as error handling and logging infrastructure.
"""

from .exceptions import (
    AlgorithmError,
    AutoTunerError,
    ConfigurationError,
    DistributedExecutionError,
    EvaluationError,
    FileOperationError,
    MetricsError,
    OpenROADError,
    ParameterError,
    ValidationError,
)
from .logging import AutoTunerLogger, get_logger

__version__ = "0.0.1"
