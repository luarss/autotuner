"""Unit tests for logging configuration."""

import json
import logging
import os
import tempfile
from unittest.mock import patch

from autotuner.core.logging import AutoTunerLogger, JSONFormatter, get_logger


class TestJSONFormatter:
    """Test cases for JSONFormatter."""

    def test_format_basic_record(self):
        """Test formatting a basic log record."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        record.module = "test"
        record.funcName = "test_function"

        result = formatter.format(record)
        parsed = json.loads(result)

        assert parsed["level"] == "INFO"
        assert parsed["logger"] == "test.logger"
        assert parsed["message"] == "Test message"
        assert parsed["module"] == "test"
        assert parsed["function"] == "test_function"
        assert parsed["line"] == 10
        assert "timestamp" in parsed

    def test_format_with_exception(self):
        """Test formatting a log record with exception."""
        formatter = JSONFormatter()

        try:
            raise ValueError("Test exception")
        except ValueError:
            import sys

            exc_info = sys.exc_info()
            record = logging.LogRecord(
                name="test.logger",
                level=logging.ERROR,
                pathname="test.py",
                lineno=10,
                msg="Error occurred",
                args=(),
                exc_info=exc_info,
            )
            record.module = "test"
            record.funcName = "test_function"

            result = formatter.format(record)
            parsed = json.loads(result)

            assert parsed["level"] == "ERROR"
            assert parsed["message"] == "Error occurred"
            assert "exception" in parsed
            assert "ValueError" in parsed["exception"]

    def test_format_with_extra_fields(self):
        """Test formatting with extra fields."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        record.module = "test"
        record.funcName = "test_function"
        record.extra_fields = {"user_id": 123, "request_id": "abc"}

        result = formatter.format(record)
        parsed = json.loads(result)

        assert parsed["user_id"] == 123
        assert parsed["request_id"] == "abc"


class TestAutoTunerLogger:
    """Test cases for AutoTunerLogger."""

    def test_singleton_pattern(self):
        """Test that AutoTunerLogger is a singleton."""
        logger1 = AutoTunerLogger()
        logger2 = AutoTunerLogger()
        assert logger1 is logger2

    def test_get_logger(self):
        """Test getting a logger."""
        logger = AutoTunerLogger.get_logger("test.module")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test.module"

    def test_configure_logging_level(self):
        """Test configuring logging level."""
        logger_instance = AutoTunerLogger()
        logger_instance.configure(level="DEBUG")

        root_logger = logging.getLogger()
        assert root_logger.level == logging.DEBUG

    def test_configure_with_json(self):
        """Test configuring with JSON format."""
        logger_instance = AutoTunerLogger()
        logger_instance.configure(use_json=True)

        root_logger = logging.getLogger()
        assert len(root_logger.handlers) > 0

        # Check if at least one handler uses JSONFormatter
        has_json_formatter = any(isinstance(handler.formatter, JSONFormatter) for handler in root_logger.handlers)
        assert has_json_formatter

    def test_configure_with_log_file(self):
        """Test configuring with log file."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            log_file = tmp_file.name

        try:
            logger_instance = AutoTunerLogger()
            logger_instance.configure(log_file=log_file)

            root_logger = logging.getLogger()

            # Check if file handler was added
            file_handlers = [handler for handler in root_logger.handlers if isinstance(handler, logging.FileHandler)]
            assert len(file_handlers) > 0

            # Test logging to file
            test_logger = logging.getLogger("test")
            test_logger.info("Test message")

            # Check if file was created and contains log
            assert os.path.exists(log_file)
            with open(log_file) as f:
                content = f.read()
                assert "Test message" in content
        finally:
            if os.path.exists(log_file):
                os.unlink(log_file)

    def test_log_with_context(self):
        """Test logging with context."""
        logger = logging.getLogger("test")

        # Mock the logger to capture the record
        with patch.object(logger, "handle") as mock_handle:
            AutoTunerLogger.log_with_context(logger, logging.INFO, "Test message", user_id=123, action="test")

            mock_handle.assert_called_once()
            record = mock_handle.call_args[0][0]

            assert record.extra_fields == {"user_id": 123, "action": "test"}
            assert record.getMessage() == "Test message"


class TestConvenienceFunctions:
    """Test cases for convenience functions."""

    def test_get_logger_function(self):
        """Test get_logger convenience function."""
        logger = get_logger("test.module")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test.module"

    def test_multiple_loggers(self):
        """Test getting multiple loggers."""
        logger1 = get_logger("module1")
        logger2 = get_logger("module2")

        assert logger1.name == "module1"
        assert logger2.name == "module2"
        assert logger1 is not logger2


class TestLoggingIntegration:
    """Integration tests for logging."""

    def test_logging_with_exceptions(self):
        """Test logging with custom exceptions."""
        from autotuner.core.exceptions import ConfigurationError

        logger = get_logger("test")

        with patch.object(logger, "error") as mock_error:
            try:
                raise ConfigurationError("Test config error")
            except ConfigurationError:
                logger.error("Configuration error occurred", exc_info=True)

            mock_error.assert_called_once()
            args, kwargs = mock_error.call_args
            assert "Configuration error occurred" in args[0]
            assert kwargs.get("exc_info") is True

    def test_structured_logging(self):
        """Test structured logging with JSON format."""
        logger_instance = AutoTunerLogger()
        logger_instance.configure(use_json=True)

        logger = get_logger("test")

        with patch("sys.stdout.write") as mock_write:
            AutoTunerLogger.log_with_context(
                logger, logging.INFO, "Structured log message", component="test", version="1.0.0"
            )

            # Check if JSON was written
            mock_write.assert_called()
            written_content = "".join(call[0][0] for call in mock_write.call_args_list)

            # Should contain JSON structure
            assert "component" in written_content
            assert "version" in written_content
