import logging
import logging.config
from .settings import LOG_FILE_PATH

log_config = {
    "version": 1,
    "formatters": {
        "verbose": {
            "format": "[%(asctime)s %(filename)s->%(funcName)s():%(lineno)s]%(levelname)s: %(message)s",
            "style": "%",
        },
        "simple": {
            "format": "%(levelname)s %(message)s",
            "style": "%",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "ERROR",
            "formatter": "simple",
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "maxBytes": 1000 * 1000,
            "filename": LOG_FILE_PATH,
            "level": "INFO",
            "formatter": "verbose",
        },
    },
    "root": {
        "level": "DEBUG",
        "handlers": ["console", "file"],
    },
}

logging.config.dictConfig(log_config)
logger = logging.getLogger(__name__)
