import os
import logging

from .utils import create_logger, logger_path

_tracepoint_initialized = False

class TracePointFormatter(logging.Formatter):
    def __init__(self):
        self.pid = os.getenv('GLOBAL_RANK', -1)  # global rank
        self.tid = 0

    def format(self, record: logging.LogRecord):
        # the chrome trace metirc include
        #   name : event name
        #   cat  : the event categories
        #   ph   : the event type, B-begin E-end
        #   ts   : tracing clock timestamp of the event, mciro sencond
        #   pid  : pid, or global rank
        #   tid  : tid, or stream id

        # must use like below:
        #    logger.info("event_name", extra={"cat": "cat", "pid": pid, "tid": tid, ph: "B"})
        try:
            format_str = ",".join(
                [
                    str(int(record.created * 1000000)),  # microsecond
                    str(self.pid),  # global rank
                    str(self.tid),  # stream_id, default is cpu
                    record.cat,
                    record.getMessage(),
                    record.ph,
                ]
            )
        except Exception as e:
            format_str = f"error logger format : {str(e)}"

        return format_str


class TracePoint:
    def __init__(self, event_name: str, cat_name: str, gpu: bool = False):
        self.logger = logging.getLogger("TracePoint")
        self.name = event_name
        self.cat = cat_name
        self.gpu = gpu

    def begin(self):
        self.record(self.name, self.cat, "B")

    def end(self):
        self.record(self.name, self.cat, "E")

    def record(self, event_name: str, cat_name: str, ph: str):
        self.logger.info(
            event_name,
            extra={
                "cat": cat_name,
                "ph": ph,
            },
        )

    def __enter__(self):
        self.begin()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end()
        return False



def tracepoint_module_setup():
    global _tracepoint_initialized
    if _tracepoint_initialized:
        return

    default_formatter = logging.Formatter(
        fmt="[%(levelname)s][%(process)d][%(name)s][%(asctime)s] %(message)s"
    )

    create_logger(logger_path, "TracePoint", TracePointFormatter())

    _tracepoint_initialized = True