"""
Lightweight logging utilities for idealist package.

Provides a simple logging interface that can be enabled/disabled by users.
"""

import logging
import sys
from typing import Optional


def setup_logger(
    name: str = "idealist",
    level: int = logging.INFO,
    enable: bool = True
) -> logging.Logger:
    """
    Set up a logger for the idealist package.

    Parameters
    ----------
    name : str, default="idealist"
        Logger name
    level : int, default=logging.INFO
        Logging level (logging.DEBUG, logging.INFO, etc.)
    enable : bool, default=True
        Whether to enable logging output

    Returns
    -------
    logging.Logger
        Configured logger instance

    Examples
    --------
    >>> logger = setup_logger(level=logging.DEBUG)
    >>> logger.info("Model fitting started")
    """
    logger = logging.getLogger(name)

    # Remove existing handlers to avoid duplicates
    logger.handlers = []

    if enable:
        logger.setLevel(level)

        # Create console handler with formatting
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)

        # Create formatter
        formatter = logging.Formatter(
            '%(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)

        logger.addHandler(handler)
    else:
        # Disable logging by setting to CRITICAL+1
        logger.setLevel(logging.CRITICAL + 1)

    # Prevent propagation to root logger
    logger.propagate = False

    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger instance.

    Parameters
    ----------
    name : str, optional
        Logger name. If None, returns the root idealist logger.

    Returns
    -------
    logging.Logger
        Logger instance
    """
    if name is None:
        return logging.getLogger("idealist")
    return logging.getLogger(f"idealist.{name}")


# Create default logger (disabled by default to not spam users)
_default_logger = setup_logger(enable=False)


__all__ = ['setup_logger', 'get_logger']
