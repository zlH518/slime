import os
import ray
import time
import logging
import torch

from ray.util.placement_group import placement_group, PlacementGroupSchedulingStrategy

from .utils import torch_available, logger_path, create_logger

def launch_memtrace_group(num_gpus, interval=1.0):
    bundles = [{"GPU": 1, "CPU": 1} for _ in range(num_gpus)]
    pg = placement_group(bundles, strategy="PACK")
    ray.get(pg.ready())

    memtrace_actors = []
    for i in range(num_gpus):
        actor = MemTrace.options(
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=pg,
                placement_group_bundle_index=i,
            )
        ).remote()
        memtrace_actors.append(actor)

    ray.util.remove_placement_group(pg)

    for actor in memtrace_actors:
        actor.run_background.remote(interval)

    return memtrace_actors


class MemTraceFormatter(logging.Formatter):
    def __init__(self):
        self.pid = os.getenv("RANK", -1)  

    def format(self, record: logging.LogRecord):
        try:
            format_str = ",".join(
                [
                    str(int(record.created * 1000000)),  
                    str(self.pid),  
                    record.getMessage(),  
                ]
            )
        except Exception as e:
            format_str = f"error logger format : {str(e)}"

        return format_str


@ray.remote(num_cpus=1)
class MemTrace:
    _logger = None
    _is_cuda_env = None

    def __init__(self):
        self._logger = logging.getLogger("MemTrace")
        create_logger(logger_path, "MemTrace", MemTraceFormatter())

    def record(self):
        if not torch_available():
            return
        free, _ = torch.cuda.mem_get_info(torch.cuda.current_device())

        self._logger.info(f"{free}")
    
    def run_background(self, interval=1.0):
        while not self._stop:
            self.record()
            time.sleep(interval)

    def stop(self):
        self._stop = True
