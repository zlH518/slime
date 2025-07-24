import os
import queue
import atexit
import logging
from pathlib import Path
import logging.handlers



_ROTATE_FILE_COUNT = 5
_ROTATE_FILE_MAX_SIZE = 200 * 1024 * 1024

def __create_async_rotating_file_handler(
    log_file_path: Path, formatter: logging.Formatter
):
    # set up handler
    # rorate to 5 file, and each file 200MB
    file_handler = logging.handlers.RotatingFileHandler(
        filename=log_file_path,
        mode="a",
        maxBytes=_ROTATE_FILE_MAX_SIZE,
        backupCount=_ROTATE_FILE_COUNT,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    queue_buffer = queue.Queue()

    # queue_handler -> listener -> file_hander
    queue_handler = logging.handlers.QueueHandler(queue_buffer)

    queue_listener = logging.handlers.QueueListener(
        queue_buffer, file_handler, respect_handler_level=True
    )
    queue_listener.start()
    atexit.register(queue_listener.stop)

    return queue_handler


def create_logger(log_root_dir: str, logger_name: str, formatter: logging.Formatter):
    log_dir = Path(log_root_dir) / logger_name
    log_dir.mkdir(parents=True, exist_ok=True)

    # set logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.addHandler(
        __create_async_rotating_file_handler(
            log_dir / f"rank_{os.getenv('GLOBAL_RANK', -1)}.log",
            formatter,
        )
    )
    logger.propagate = False