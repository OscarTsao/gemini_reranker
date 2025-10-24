"""Logging utilities using rich tracebacks."""

from __future__ import annotations

import logging
from typing import Optional

from rich.logging import RichHandler


def init_logger(level: int = logging.INFO) -> None:
    """Initialise global logging configuration."""
    if logging.getLogger().handlers:
        return

    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, markup=False)],
    )


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Get a configured logger."""
    init_logger()
    return logging.getLogger(name if name else __name__)
