"""Logging configuration for AutoTuner."""

import json
import logging
import sys
from datetime import datetime
from typing import Any, Optional


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        # Add extra fields if present
        if hasattr(record, "extra_fields"):
            log_entry.update(record.extra_fields)

        return json.dumps(log_entry)


class AutoTunerLogger:
    """Centralized logging configuration for AutoTuner."""

    _instance: Optional["AutoTunerLogger"] = None
    _initialized = False

    def __new__(cls) -> "AutoTunerLogger":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self._setup_logging()
            self._initialized = True

    def _setup_logging(self):
        """Set up logging configuration."""
        # Get root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)

        # Clear existing handlers
        root_logger.handlers.clear()

        # Create console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)

        # Create file handler
        file_handler = logging.FileHandler("autotuner.log")
        file_handler.setLevel(logging.DEBUG)

        # Set formatters
        console_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        json_formatter = JSONFormatter()

        console_handler.setFormatter(console_formatter)
        file_handler.setFormatter(json_formatter)

        # Add handlers to root logger
        root_logger.addHandler(console_handler)
        root_logger.addHandler(file_handler)

    def configure(
        self,
        level: str = "INFO",
        use_json: bool = False,
        log_file: str | None = None,
        extra_config: dict[str, Any] | None = None,
    ):
        """Configure logging with custom settings."""
        # Set logging level
        numeric_level = getattr(logging, level.upper(), logging.INFO)
        logging.getLogger().setLevel(numeric_level)

        # Update handlers based on configuration
        root_logger = logging.getLogger()

        # Clear existing handlers
        root_logger.handlers.clear()

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)

        if use_json:
            console_formatter = JSONFormatter()
        else:
            console_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)

        # File handler
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(JSONFormatter())
            root_logger.addHandler(file_handler)

        # Apply extra configuration
        if extra_config:
            for key, value in extra_config.items():
                if hasattr(logging, key.upper()):
                    setattr(logging, key.upper(), value)

    @staticmethod
    def get_logger(name: str) -> logging.Logger:
        """Get a logger with the specified name."""
        return logging.getLogger(name)

    @staticmethod
    def log_with_context(logger: logging.Logger, level: int, message: str, **context: Any):
        """Log a message with additional context."""
        record = logger.makeRecord(logger.name, level, "", 0, message, (), None)
        record.extra_fields = context
        logger.handle(record)


# Initialize the logger
logger_instance = AutoTunerLogger()


# Convenience function for getting loggers
def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name."""
    return AutoTunerLogger.get_logger(name)
