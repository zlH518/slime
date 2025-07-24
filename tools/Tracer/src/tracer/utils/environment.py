
_ROOT_LOGGER_PATH = "/volume/pt-train/users/mingjie/hzl_code/code/slime/experiments/log/TracePoint"

def torch_available():
    """
    Check if PyTorch is available in the environment.
    
    Returns:
        bool: True if PyTorch is available, False otherwise.
    """
    try:
        import torch
        import torch.distributed

        return True
    except ImportError:
        return False


