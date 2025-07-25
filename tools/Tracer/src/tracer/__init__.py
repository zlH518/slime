import logging

from tracer.tracepoint import TracePoint, TracePointFormatter
from tracer.memTrace import MemTracePoint, MemTracePointFormatter
from .utils import create_logger, logger_path

_vinit_initialized = False

def _tracepoint_module_setup():
    default_formatter = logging.Formatter(
        fmt="[%(levelname)s][%(process)d][%(name)s][%(asctime)s] %(message)s"
    )

    create_logger(logger_path, "TracePoint", TracePointFormatter())
    create_logger(logger_path, "MemTracePoint", MemTracePointFormatter())


def vinit():
    global _vinit_initialized
    if _vinit_initialized:
        return

    _tracepoint_module_setup()
    MemTracePoint.initialize()
    _vinit_initialized = True

__all__ = [
    "vinit",
    "TracePoint",
    "MemTracePoint"
]