import os
import ray
import time
import logging
import torch

from .utils import torch_available, logger_path, create_logger


class MemTracePointFormatter(logging.Formatter):
    def __init__(self):
        self.pid = os.getenv("GLOBAL_RANK", -1)  

    def format(self, record: logging.LogRecord):
        try:
            format_str = ",".join(
                [
                    str(int(record.created * 1000000)),
                    str(self.pid),
                    record.getMessage(),
                    str(record.used),
                ]
            )
        except Exception as e:
            format_str = f"error logger format : {str(e)}"

        return format_str


class MemTracePoint:
    _logger = None
    _is_cuda_env = None

    @classmethod
    def initialize(cls):
        cls._logger = logging.getLogger("MemTracePoint")
        try:
            import torch

            cls._is_cuda_env = torch.cuda.is_available()
        except Exception:
            cls._is_cuda_env = False

    @staticmethod
    def record(event_name: str):
        if not MemTracePoint._is_cuda_env:
            return
            
        import torch

        free, total = torch.cuda.mem_get_info(torch.cuda.current_device())

        MemTracePoint._logger.info(
            event_name,
            extra={
                "used": total-free,
            }
        )
