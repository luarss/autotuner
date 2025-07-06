"""Unit tests for custom exceptions."""

import pytest

from autotuner.core.exceptions import (
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


class TestAutoTunerError:
    """Test cases for AutoTunerError base class."""

    def test_basic_exception(self):
        """Test basic exception creation."""
        error = AutoTunerError("Test error")
        assert str(error) == "Test error"
        assert error.message == "Test error"
        assert error.details == {}

    def test_exception_with_details(self):
        """Test exception with details."""
        details = {"key": "value", "count": 42}
        error = AutoTunerError("Test error", details)
        assert str(error) == "Test error"
        assert error.message == "Test error"
        assert error.details == details

    def test_exception_inheritance(self):
        """Test that AutoTunerError inherits from Exception."""
        error = AutoTunerError("Test error")
        assert isinstance(error, Exception)


class TestSpecificExceptions:
    """Test cases for specific exception types."""

    def test_configuration_error(self):
        """Test ConfigurationError."""
        error = ConfigurationError("Config error")
        assert isinstance(error, AutoTunerError)
        assert str(error) == "Config error"

    def test_algorithm_error(self):
        """Test AlgorithmError."""
        error = AlgorithmError("Algorithm error")
        assert isinstance(error, AutoTunerError)
        assert str(error) == "Algorithm error"

    def test_evaluation_error(self):
        """Test EvaluationError."""
        error = EvaluationError("Evaluation error")
        assert isinstance(error, AutoTunerError)
        assert str(error) == "Evaluation error"

    def test_distributed_execution_error(self):
        """Test DistributedExecutionError."""
        error = DistributedExecutionError("Distributed error")
        assert isinstance(error, AutoTunerError)
        assert str(error) == "Distributed error"

    def test_openroad_error(self):
        """Test OpenROADError."""
        error = OpenROADError("OpenROAD error")
        assert isinstance(error, AutoTunerError)
        assert str(error) == "OpenROAD error"

    def test_metrics_error(self):
        """Test MetricsError."""
        error = MetricsError("Metrics error")
        assert isinstance(error, AutoTunerError)
        assert str(error) == "Metrics error"

    def test_parameter_error(self):
        """Test ParameterError."""
        error = ParameterError("Parameter error")
        assert isinstance(error, AutoTunerError)
        assert str(error) == "Parameter error"

    def test_file_operation_error(self):
        """Test FileOperationError."""
        error = FileOperationError("File error")
        assert isinstance(error, AutoTunerError)
        assert str(error) == "File error"

    def test_validation_error(self):
        """Test ValidationError."""
        error = ValidationError("Validation error")
        assert isinstance(error, AutoTunerError)
        assert str(error) == "Validation error"


class TestExceptionRaising:
    """Test cases for raising exceptions."""

    def test_raise_autotuner_error(self):
        """Test raising AutoTunerError."""
        with pytest.raises(AutoTunerError, match="Test error"):
            raise AutoTunerError("Test error")

    def test_raise_configuration_error(self):
        """Test raising ConfigurationError."""
        with pytest.raises(ConfigurationError, match="Config error"):
            raise ConfigurationError("Config error")

    def test_raise_algorithm_error(self):
        """Test raising AlgorithmError."""
        with pytest.raises(AlgorithmError, match="Algorithm error"):
            raise AlgorithmError("Algorithm error")

    def test_catch_as_base_exception(self):
        """Test catching specific exception as base exception."""
        with pytest.raises(AutoTunerError):
            raise ConfigurationError("Config error")

    def test_catch_as_standard_exception(self):
        """Test catching as standard Exception."""
        with pytest.raises(Exception):  # noqa: B017
            raise AutoTunerError("Test error")
