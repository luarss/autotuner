"""Base algorithm implementations for AutoTuner."""

import random
from abc import ABC
from collections import namedtuple
from typing import Any

import numpy as np
import torch
from ax.service.ax_client import AxClient
from ray.tune.schedulers import PopulationBasedTraining
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.ax import AxSearch
from ray.tune.search.basic_variant import BasicVariantGenerator
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.search.optuna import OptunaSearch

from autotuner.core.interfaces import Algorithm


class BaseAlgorithm(Algorithm, ABC):
    """Base implementation for all algorithms."""

    def __init__(self, seed: int | None = None):
        self.seed = seed
        self.search_space: dict[str, Any] = {}
        self.results: list[dict[str, Any]] = []
        self.best_config: dict[str, Any] = {}
        self._algorithm = None

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

    def set_search_space(self, search_space: dict[str, Any]) -> None:
        """Set the search space for the algorithm."""
        self.search_space = search_space

    def get_best_config(self) -> dict[str, Any]:
        """Get the best configuration found by the algorithm."""
        return self.best_config

    def get_results(self) -> list[dict[str, Any]]:
        """Get all results from the algorithm execution."""
        return self.results


class HyperOptAlgorithm(BaseAlgorithm):
    """HyperOpt algorithm implementation."""

    def __init__(self, seed: int | None = None, points_to_evaluate: list[dict[str, Any]] | None = None):
        super().__init__(seed)
        self.points_to_evaluate = points_to_evaluate or []

    def run(self, _config: dict[str, Any]) -> Any:
        """Execute the HyperOpt algorithm with the given configuration."""
        self._algorithm = HyperOptSearch(
            points_to_evaluate=self.points_to_evaluate,
            random_state_seed=self.seed,
        )
        return self._algorithm


class AxAlgorithm(BaseAlgorithm):
    """Ax algorithm implementation."""

    def __init__(
        self,
        seed: int | None = None,
        points_to_evaluate: list[dict[str, Any]] | None = None,
        experiment_name: str = "autotuner_experiment",
        metric_name: str = "score",
    ):
        super().__init__(seed)
        self.points_to_evaluate = points_to_evaluate or []
        self.experiment_name = experiment_name
        self.metric_name = metric_name

    def run(self, _config: dict[str, Any]) -> Any:
        """Execute the Ax algorithm with the given configuration."""
        ax_client = AxClient(
            enforce_sequential_optimization=False,
            random_seed=self.seed,
        )
        AxClientMetric = namedtuple("AxClientMetric", "minimize")
        ax_client.create_experiment(
            name=self.experiment_name,
            parameters=self.search_space,
            objectives={self.metric_name: AxClientMetric(minimize=True)},
        )
        self._algorithm = AxSearch(ax_client=ax_client, points_to_evaluate=self.points_to_evaluate)
        return self._algorithm


class OptunaAlgorithm(BaseAlgorithm):
    """Optuna algorithm implementation."""

    def __init__(self, seed: int | None = None, points_to_evaluate: list[dict[str, Any]] | None = None):
        super().__init__(seed)
        self.points_to_evaluate = points_to_evaluate or []

    def run(self, _config: dict[str, Any]) -> Any:
        """Execute the Optuna algorithm with the given configuration."""
        self._algorithm = OptunaSearch(points_to_evaluate=self.points_to_evaluate, seed=self.seed)
        return self._algorithm


class PBTAlgorithm(BaseAlgorithm):
    """Population Based Training algorithm implementation."""

    def __init__(self, perturbation_interval: int = 10, hyperparam_mutations: dict[str, Any] | None = None):
        super().__init__(None)  # PBT doesn't support seeds
        self.perturbation_interval = perturbation_interval
        self.hyperparam_mutations = hyperparam_mutations or {}

    def run(self, _config: dict[str, Any]) -> Any:
        """Execute the PBT algorithm with the given configuration."""
        self._algorithm = PopulationBasedTraining(
            time_attr="training_iteration",
            perturbation_interval=self.perturbation_interval,
            hyperparam_mutations=self.hyperparam_mutations,
            synch=True,
        )
        return self._algorithm


class RandomAlgorithm(BaseAlgorithm):
    """Random/Grid search algorithm implementation."""

    def __init__(self, seed: int | None = None, max_concurrent: int = 1):
        super().__init__(seed)
        self.max_concurrent = max_concurrent

    def run(self, _config: dict[str, Any]) -> Any:
        """Execute the Random algorithm with the given configuration."""
        self._algorithm = BasicVariantGenerator(
            max_concurrent=self.max_concurrent,
            random_state=self.seed,
        )
        return self._algorithm


def create_algorithm(algorithm_name: str, **kwargs) -> Algorithm:
    """Factory function to create algorithm instances.

    Args:
        algorithm_name: Name of the algorithm to create
        **kwargs: Additional arguments for the algorithm

    Returns:
        Algorithm instance

    Raises:
        ValueError: If algorithm name is not supported
    """
    algorithms = {
        "hyperopt": HyperOptAlgorithm,
        "ax": AxAlgorithm,
        "optuna": OptunaAlgorithm,
        "pbt": PBTAlgorithm,
        "random": RandomAlgorithm,
    }

    if algorithm_name not in algorithms:
        raise ValueError(f"Unknown algorithm {algorithm_name}. Supported algorithms: {', '.join(algorithms.keys())}")

    return algorithms[algorithm_name](**kwargs)


def wrap_with_concurrency_limiter(algorithm: Any, max_concurrent: int) -> Any:
    """Wrap algorithm with concurrency limiter if needed.

    Args:
        algorithm: Algorithm to wrap
        max_concurrent: Maximum number of concurrent trials

    Returns:
        Wrapped algorithm or original if not applicable
    """
    if not isinstance(algorithm, BasicVariantGenerator | PopulationBasedTraining):
        return ConcurrencyLimiter(algorithm, max_concurrent=max_concurrent)
    return algorithm
