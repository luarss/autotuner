"""Core interfaces and abstract base classes for AutoTuner components."""

from abc import ABC, abstractmethod
from typing import Any


class Algorithm(ABC):
    """Abstract base class for all hyperparameter optimization algorithms."""

    @abstractmethod
    def run(self, config: dict[str, Any]) -> Any:
        """Execute the algorithm with the given configuration.

        Args:
            config: Configuration dictionary containing algorithm parameters

        Returns:
            Result of the algorithm execution
        """
        pass

    @abstractmethod
    def set_search_space(self, search_space: dict[str, Any]) -> None:
        """Set the search space for the algorithm.

        Args:
            search_space: Dictionary defining the parameter search space
        """
        pass

    @abstractmethod
    def get_best_config(self) -> dict[str, Any]:
        """Get the best configuration found by the algorithm.

        Returns:
            Dictionary containing the best parameter configuration
        """
        pass

    @abstractmethod
    def get_results(self) -> list[dict[str, Any]]:
        """Get all results from the algorithm execution.

        Returns:
            List of dictionaries containing all trial results
        """
        pass


class Evaluator(ABC):
    """Abstract base class for all evaluation functions."""

    @abstractmethod
    def evaluate(self, config: dict[str, Any]) -> dict[str, float | int]:
        """Evaluate a configuration and return metrics.

        Args:
            config: Configuration dictionary to evaluate

        Returns:
            Dictionary containing evaluation metrics
        """
        pass

    @abstractmethod
    def set_metric(self, metric_name: str, mode: str = "min") -> None:
        """Set the primary metric for evaluation.

        Args:
            metric_name: Name of the metric to optimize
            mode: Optimization mode ("min" or "max")
        """
        pass

    @abstractmethod
    def get_evaluation_results(self) -> list[dict[str, Any]]:
        """Get all evaluation results.

        Returns:
            List of dictionaries containing all evaluation results
        """
        pass

    @abstractmethod
    def setup_evaluation_environment(self) -> None:
        """Set up the evaluation environment."""
        pass


class Runner(ABC):
    """Abstract base class for all execution runners."""

    @abstractmethod
    def execute(self, command: str, **kwargs) -> dict[str, Any]:
        """Execute a command and return results.

        Args:
            command: Command to execute
            **kwargs: Additional execution parameters

        Returns:
            Dictionary containing execution results
        """
        pass

    @abstractmethod
    def setup_execution_context(self, context: dict[str, Any]) -> None:
        """Set up the execution context.

        Args:
            context: Dictionary containing execution context parameters
        """
        pass

    @abstractmethod
    def get_execution_logs(self) -> list[str]:
        """Get execution logs.

        Returns:
            List of log messages from execution
        """
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """Clean up resources after execution."""
        pass
