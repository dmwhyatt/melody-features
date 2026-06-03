"""Logging helpers for melody feature extraction."""

import logging


def _setup_logger(level: int = logging.INFO) -> logging.Logger:
    """Set up and configure the logger for the melodic feature set."""
    logger = logging.getLogger("melody_features")
    logger.setLevel(level)

    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
        )
        console_handler.setFormatter(formatter)

        logger.addHandler(console_handler)

    return logger
