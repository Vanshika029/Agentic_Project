import logging
import sys

def get_logger(name: str = "enterprise_hr_ai") -> logging.Logger:
    """Configures and returns a structured application logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        
        # Console Handler
        c_handler = logging.StreamHandler(sys.stdout)
        c_handler.setLevel(logging.INFO)
        
        # Formatter matching enterprise log specification
        # Format: 2026-08-27 10:30:15 | INFO | Message
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-5s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        c_handler.setFormatter(formatter)
        logger.addHandler(c_handler)
        logger.propagate = False
        
    return logger

logger = get_logger()
