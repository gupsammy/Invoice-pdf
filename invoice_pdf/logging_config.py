"""Logging configuration for Invoice PDF processing."""
import logging
import logging.config
from pathlib import Path
from typing import Dict, Any


def get_logging_config(logs_folder: Path, log_filename: str = "invoice_extraction_2step_enhanced.log") -> Dict[str, Any]:
    """Get logging configuration dictionary."""
    logs_folder.mkdir(parents=True, exist_ok=True)
    log_file_path = logs_folder / log_filename
    
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s - %(levelname)s - %(message)s"
            },
            "detailed": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": "INFO",
                "formatter": "standard",
                "stream": "ext://sys.stdout"
            },
            "file": {
                "class": "logging.FileHandler",
                "level": "INFO",
                "formatter": "detailed",
                "filename": str(log_file_path),
                "mode": "a",
                "encoding": "utf-8"
            }
        },
        "root": {
            "level": "INFO",
            "handlers": ["console", "file"]
        },
        "loggers": {
            "invoice_pdf": {
                "level": "DEBUG",
                "handlers": ["console", "file"],
                "propagate": False
            }
        }
    }


def setup_logging(logs_folder: Path, log_filename: str = "invoice_extraction_2step_enhanced.log") -> None:
    """Set up logging with the specified configuration."""
    config = get_logging_config(logs_folder, log_filename)
    
    # Clear any existing handlers to prevent duplicate logs
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    logging.config.dictConfig(config)