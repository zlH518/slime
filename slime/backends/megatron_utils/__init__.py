import logging
import torch

try:
    from torch_memory_saver import torch_memory_saver
    import deep_ep

    old_init = deep_ep.Buffer.__init__

    def new_init(self, *args, **kwargs):
        torch_memory_saver._impl._binary_wrapper.cdll.tms_set_interesting_region(False)
        old_init(self, *args, **kwargs)
        torch.cuda.synchronize()
        torch_memory_saver._impl._binary_wrapper.cdll.tms_set_interesting_region(True)

    deep_ep.Buffer.__init__ = new_init
except ImportError:
    logging.warning("deep_ep is not installed, some functionalities may be limited.")


from .actor import MegatronTrainRayActor
from .arguments import parse_args, validate_args, set_default_megatron_args
from .checkpoint import load_checkpoint, save_checkpoint
from .initialize import init
from .model import initialize_model_and_optimizer

logging.getLogger().setLevel(logging.WARNING)


__all__ = [
    "parse_args",
    "validate_args",
    "load_checkpoint",
    "save_checkpoint",
    "set_default_megatron_args",
    "MegatronTrainRayActor",
    "init",
    "initialize_model_and_optimizer",
]
