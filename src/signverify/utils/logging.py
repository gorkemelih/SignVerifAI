"""Structured logging setup for SignVerify.

Provides consistent logging across all modules with Rich formatting.
Prevents duplicate log entries by tracking initialized loggers.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler

# Track which loggers have been set up to avoid duplicates
_loggers_initialized = set()


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
    module_name: str = "signverify",
) -> logging.Logger:
    """
    Set up structured logging with Rich console output.
    Only configures the logger once to prevent duplicate handlers.
    """
    logger = logging.getLogger(module_name)
    
    # Only set up once per logger name
    if module_name in _loggers_initialized:
        return logger
    
    logger.setLevel(level)
    
    # Clear existing handlers to prevent duplicates
    logger.handlers.clear()
    
    # Prevent propagation to root logger (prevents duplicates)
    logger.propagate = False
    
    # Console handler with Rich
    console = Console(stderr=True)
    console_handler = RichHandler(
        console=console,
        show_time=True,
        show_path=False,
        rich_tracebacks=True,
        markup=True,
    )
    console_handler.setLevel(level)
    console_format = logging.Formatter("%(message)s")
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_format = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
    
    _loggers_initialized.add(module_name)
    
    return logger


def get_logger(name: str = "signverify") -> logging.Logger:
    """Get existing logger or create new one."""
    logger = logging.getLogger(name)
    if name not in _loggers_initialized:
        setup_logging(module_name=name)
    return logger
