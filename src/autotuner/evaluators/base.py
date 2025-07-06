"""Base evaluator implementations for AutoTuner."""

from abc import ABC
from typing import Any

from autotuner.core.interfaces import Evaluator


class BaseEvaluator(Evaluator, ABC):
    """Base evaluator implementation."""

    def __init__(self, metric_name: str = "score", mode: str = "min"):
        self.metric_name = metric_name
        self.mode = mode
        self.evaluation_results: list[dict[str, Any]] = []

    def set_metric(self, metric_name: str, mode: str = "min") -> None:
        """Set the primary metric for evaluation.

        Args:
            metric_name: Name of the metric to optimize
            mode: Optimization mode ("min" for minimize, "max" for maximize)
        """
        self.metric_name = metric_name
        self.mode = mode

    def get_evaluation_results(self) -> list[dict[str, Any]]:
        """Get all evaluation results."""
        return self.evaluation_results.copy()

    def setup_evaluation_environment(self) -> None:
        """Set up the evaluation environment."""
        pass


class DefaultEvaluator(BaseEvaluator):
    """Default evaluator that optimizes effective clock period."""

    def __init__(self, error_metric: float = 1e9):
        super().__init__(metric_name="effective_clk_period", mode="min")
        self.error_metric = error_metric

    def evaluate(self, config: dict[str, Any]) -> dict[str, float | int]:
        """Evaluate a configuration and return metrics."""
        # This is a placeholder - actual evaluation would run OpenROAD flow
        # For now, return dummy metrics
        metrics = {"clk_period": 2.0, "worst_slack": -0.1, "num_drc": 0, "effective_clk_period": 2.1}

        # Check for errors
        error = "ERR" in str(config.values())
        not_found = "N/A" in str(config.values())

        if error or not_found:
            metrics = {
                "clk_period": self.error_metric,
                "worst_slack": 0,
                "num_drc": self.error_metric,
                "effective_clk_period": self.error_metric,
            }

        # Store evaluation result
        result = {
            "config": config,
            "metrics": metrics,
            "timestamp": None,  # Could add timestamp if needed
        }
        self.evaluation_results.append(result)

        return metrics


class PPAEvaluator(BaseEvaluator):
    """PPA (Performance, Power, Area) improvement evaluator."""

    def __init__(self, reference_dict: dict[str, Any], error_metric: float = 1e9):
        super().__init__(metric_name="ppa_score", mode="min")
        self.reference_dict = reference_dict
        self.error_metric = error_metric

    def evaluate(self, config: dict[str, Any]) -> dict[str, float | int]:
        """Evaluate a configuration and return PPA metrics."""
        # This is a placeholder - actual evaluation would run OpenROAD flow
        # For now, return dummy metrics
        metrics = {"clk_period": 2.0, "worst_slack": -0.1, "total_power": 0.5, "final_util": 60.0, "num_drc": 0}

        # Check for errors
        error = "ERR" in str(config.values()) or "ERR" in str(self.reference_dict.values())
        not_found = "N/A" in str(config.values()) or "N/A" in str(self.reference_dict.values())
        ppa_score = self.error_metric if not_found or error else self._calculate_ppa(metrics)

        result_metrics = {
            **metrics,
            "ppa_score": ppa_score,
            "effective_clk_period": metrics["clk_period"] - metrics["worst_slack"],
        }

        # Store evaluation result
        result = {"config": config, "metrics": result_metrics, "timestamp": None}
        self.evaluation_results.append(result)

        return result_metrics

    def _calculate_ppa(self, metrics: dict[str, Any]) -> float:
        """Calculate PPA score based on metrics."""
        coeff_perform, coeff_power, coeff_area = 10000, 100, 100

        # Calculate effective clock period
        eff_clk_period = metrics["clk_period"]
        if metrics["worst_slack"] < 0:
            eff_clk_period -= metrics["worst_slack"]

        eff_clk_period_ref = self.reference_dict["clk_period"]
        if self.reference_dict["worst_slack"] < 0:
            eff_clk_period_ref -= self.reference_dict["worst_slack"]

        def percent(x_1: float, x_2: float) -> float:
            return (x_1 - x_2) / x_1 * 100 if x_1 != 0 else 0

        # Calculate improvements
        performance = percent(eff_clk_period_ref, eff_clk_period)
        power = percent(self.reference_dict["total_power"], metrics["total_power"])
        area = percent(100 - self.reference_dict["final_util"], 100 - metrics["final_util"])

        # Calculate PPA score (lower is better)
        ppa_upper_bound = (coeff_perform + coeff_power + coeff_area) * 100
        ppa = performance * coeff_perform + power * coeff_power + area * coeff_area

        return ppa_upper_bound - ppa

    def setup_evaluation_environment(self) -> None:
        """Set up PPA evaluation environment."""
        super().setup_evaluation_environment()
        # Could add specific PPA evaluation setup here


def create_evaluator(evaluator_name: str, **kwargs) -> Evaluator:
    """Factory function to create evaluator instances.

    Args:
        evaluator_name: Name of the evaluator to create
        **kwargs: Additional arguments for the evaluator

    Returns:
        Evaluator instance

    Raises:
        ValueError: If evaluator name is not supported
    """
    evaluators = {
        "default": DefaultEvaluator,
        "ppa-improv": PPAEvaluator,
    }

    if evaluator_name not in evaluators:
        raise ValueError(f"Unknown evaluator {evaluator_name}. Supported evaluators: {', '.join(evaluators.keys())}")

    return evaluators[evaluator_name](**kwargs)
