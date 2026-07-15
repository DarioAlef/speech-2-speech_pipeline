"""Central structured-logging setup used across the backend."""
from __future__ import annotations

import logging
import os

_CONFIGURED = False


def configure_logging(level: str | None = None) -> None:
    """Configure root logging once. Level from arg or VCL_LOG_LEVEL (default INFO)."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    level_name = (level or os.environ.get("VCL_LOG_LEVEL", "INFO")).upper()
    logging.basicConfig(
        level=getattr(logging, level_name, logging.INFO),
        format="%(asctime)s %(levelname)-7s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    return logging.getLogger(name)
