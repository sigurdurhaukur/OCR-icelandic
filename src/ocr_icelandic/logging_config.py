"""Centralized logging configuration for ocr_icelandic module.

Provides a shared parent logger with rich formatting and child loggers for each module.
Supports DEBUG level logging with environment variable override.

Usage:
    from ocr_icelandic.logging_config import get_logger

    logger = get_logger(__name__)
    logger.debug("Detailed diagnostic information")
    logger.info("High-level operation tracking")
    logger.warning("Unexpected but handled situations")
    logger.error("Operation failures")

Environment Variables:
    OCR_LOG_LEVEL: Override default log level (DEBUG, INFO, WARNING, ERROR)
                   Example: OCR_LOG_LEVEL=INFO
"""

import logging
import os

# Global lock to ensure thread-safe lazy initialization
_INIT_LOCK = None
_INITIALIZED = False


def get_logger(name: str) -> logging.Logger:
    """Get a child logger for the specified module.

    Lazily initializes the root logger on first call with rich formatting.
    All subsequent calls return child loggers that inherit the configuration.

    Args:
        name: Module name, typically __name__ of the calling module.
              Examples: "ocr_icelandic.utils.color", "ocr_icelandic.transformations.effects"

    Returns:
        logging.Logger: Configured logger for the module.

    Example:
        >>> from ocr_icelandic.logging_config import get_logger
        >>> logger = get_logger(__name__)
        >>> logger.debug("Processing image with dimensions %dx%d", width, height)
    """
    global _INITIALIZED, _INIT_LOCK

    # Lazy initialization on first call
    if not _INITIALIZED:
        _initialize_logging()
        _INITIALIZED = True

    # Return child logger for the module
    return logging.getLogger(name)


def _initialize_logging() -> None:
    """Initialize the root logger with rich handler (internal use).

    Called once on first get_logger() call. Sets up:
    - Parent logger: "ocr_icelandic"
    - Handler: RichHandler with colors, timestamps, file paths
    - Level: DEBUG (overridable via OCR_LOG_LEVEL environment variable)
    - Format: Minimal (rich handles the formatting)
    """
    try:
        from rich.logging import RichHandler
    except ImportError:
        # Fallback to standard logging if rich is not available
        _initialize_logging_fallback()
        return

    # Get log level from environment or use INFO as default
    log_level_str = os.getenv("OCR_LOG_LEVEL", "INFO").upper()
    try:
        log_level = getattr(logging, log_level_str)
    except AttributeError:
        log_level = logging.INFO

    # Create root logger
    root_logger = logging.getLogger("ocr_icelandic")
    root_logger.setLevel(log_level)

    # Remove any existing handlers to avoid duplicates
    root_logger.handlers.clear()

    # Create rich handler with useful settings
    handler = RichHandler(
        show_time=True,  # Show timestamp
        show_level=True,  # Show log level (DEBUG, INFO, etc.)
        show_path=True,  # Show file path and line number
        rich_tracebacks=True,  # Format tracebacks with rich styling
        markup=True,  # Allow [bold], [red], etc. in messages
    )
    handler.setLevel(log_level)

    # Minimal format - rich handles most of the formatting
    formatter = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
    handler.setFormatter(formatter)

    # Add handler to root logger
    root_logger.addHandler(handler)

    # Prevent propagation to avoid duplicate logs if parent loggers exist
    root_logger.propagate = False


def _initialize_logging_fallback() -> None:
    """Fallback logging setup without rich (if rich is not installed).

    Uses standard library logging with a StreamHandler.
    This should only be called if rich.logging.RichHandler is not available.
    """
    log_level_str = os.getenv("OCR_LOG_LEVEL", "DEBUG").upper()
    try:
        log_level = getattr(logging, log_level_str)
    except AttributeError:
        log_level = logging.DEBUG

    # Create root logger
    root_logger = logging.getLogger("ocr_icelandic")
    root_logger.setLevel(log_level)

    # Remove any existing handlers
    root_logger.handlers.clear()

    # Create standard handler
    handler = logging.StreamHandler()
    handler.setLevel(log_level)

    # Standard format with timestamp, level, and message
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)

    # Add handler to root logger
    root_logger.addHandler(handler)

    # Prevent propagation
    root_logger.propagate = False
