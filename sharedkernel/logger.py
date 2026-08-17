from __future__ import annotations

import logging
import sys

_LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"
_ROOT_LOGGER_NAME = "bombifikator"
_DEFAULT_LOG_LEVEL = "INFO"
_configured = False


def _configure_root() -> None:
    global _configured
    if _configured:
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    root = logging.getLogger(_ROOT_LOGGER_NAME)
    root.setLevel(_DEFAULT_LOG_LEVEL)
    root.addHandler(handler)
    root.propagate = False
    _configured = True


def set_log_level(level: str) -> None:
    _configure_root()
    logging.getLogger(_ROOT_LOGGER_NAME).setLevel(level.upper())


def get_logger(name: str) -> logging.Logger:
    _configure_root()
    return logging.getLogger(f"{_ROOT_LOGGER_NAME}.{name}")
