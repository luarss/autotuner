"""Custom exception hierarchy for AutoTuner."""


class AutoTunerError(Exception):
    """Base exception class for all AutoTuner errors."""

    def __init__(self, message: str = "", details: dict = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ConfigurationError(AutoTunerError):
    """Raised when there's an error with configuration loading or validation."""

    pass


class AlgorithmError(AutoTunerError):
    """Raised when there's an error with algorithm execution."""

    pass


class EvaluationError(AutoTunerError):
    """Raised when there's an error during evaluation."""

    pass


class DistributedExecutionError(AutoTunerError):
    """Raised when there's an error with distributed execution."""

    pass


class OpenROADError(AutoTunerError):
    """Raised when there's an error with OpenROAD flow execution."""

    pass


class MetricsError(AutoTunerError):
    """Raised when there's an error with metrics processing."""

    pass


class ParameterError(AutoTunerError):
    """Raised when there's an error with parameter handling."""

    pass


class FileOperationError(AutoTunerError):
    """Raised when there's an error with file operations."""

    pass


class ValidationError(AutoTunerError):
    """Raised when validation fails."""

    pass
