import torch
from cumem_allocator import CuMemAllocator

from slime.utils.memory_utils import clear_memory, print_memory, available_memory

def test1():
    allocator = CuMemAllocator.get_instance()

    with allocator.use_memory_pool(tag="model"):
        from transformers import AutoModel
        model = AutoModel.from_pretrained("/volume/pt-train/models/Qwen2.5-7B").cuda()

    print(f"Memory-Usage:", available_memory())

    clear_memory()
    allocator.sleep("model")
    clear_memory()
    print(f"Memory-Usage:", available_memory())

    allocator.wake_up(["model"])
    print(f"Memory-Usage:", available_memory())

def test2():
    from transformers import AutoModel
    model = AutoModel.from_pretrained("/volume/pt-train/models/Qwen3-4B").cuda()
    for name, param in model.named_parameters():
        print(name)

if __name__ == "__main__":
    test2()