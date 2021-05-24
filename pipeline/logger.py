import coloredlogs
import logging
logging.basicConfig(level=logging.DEBUG)
logging.getLogger('chardet.universaldetector').setLevel(logging.INFO)
logging.getLogger('matplotlib').setLevel(logging.INFO)
logging.getLogger('urllib3.connectionpool').setLevel(logging.INFO)
import sys
import tempfile
from logging.handlers import MemoryHandler

FORMAT = "%(asctime)s — %(levelname)s — %(module)s — %(funcName)s - %(message)s"
FORMATTER = logging.Formatter(FORMAT)
COLOR_FORMATTER = coloredlogs.ColoredFormatter(FORMAT)
LOG_FILE = tempfile.NamedTemporaryFile(delete=False)
CAPACITY = 512


def get_console_handler():
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(COLOR_FORMATTER)
    return console_handler


def get_file_handler():
    file_handler = logging.FileHandler(LOG_FILE.name)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(FORMATTER)
    return file_handler


def get_memory_handler():
    memory_handler = logging.handlers.MemoryHandler(CAPACITY, flushLevel=logging.ERROR, target=get_file_handler())
    memory_handler.setLevel(logging.DEBUG)
    memory_handler.setFormatter(FORMATTER)
    return memory_handler


def get_logger(logger_name):
    logger = logging.getLogger(logger_name)
    logger.addHandler(get_console_handler())
    logger.addHandler(get_memory_handler())
    logger.propagate = False
    return logger


log = get_logger(__name__)
