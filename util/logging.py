"""Shared logging setup for all steps and the pipeline."""

import logging


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Get a named logger with a single stream handler configured."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger
