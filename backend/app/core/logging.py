import logging
import sys
from typing import Optional

def setup_logger(name: str = "campus_rag", level: Optional[str] = None) -> logging.Logger:
    """
    Configures and returns a structured logger with standardized format.
    Suitable for 12-factor application observability.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-7s | [%(name)s:%(funcName)s:%(lineno)d] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level or logging.INFO)
        logger.propagate = False
    return logger

logger = setup_logger("campus_rag")
