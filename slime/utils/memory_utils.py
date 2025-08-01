import gc
import torch
import torch.distributed as dist


def clear_memory():
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()


def available_memory():
    free, total = torch.cuda.mem_get_info(torch.cuda.current_device())
    return {
        "gpu": str(torch.cuda.current_device()),
        "total_GB": round(total / (1024**3), 2),
        "free_GB": round(free / (1024**3), 2),
        "used_GB": round((total - free) / (1024**3), 2),
    }


def print_memory(msg):
    print("*"*200)
    print(f"Memory-Usage-rank-{dist.get_rank()}-{msg}:", available_memory())
    print("*"*200)