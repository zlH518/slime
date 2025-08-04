"""
PyTorch兼容内存池管理方案

这个模块支持三种内存管理模式：

1. PyTorch标准内存管理 (args.colocate=False, args.use_pytorch_pool=False)
   - 优点：与PyTorch分布式通信完全兼容
   - 缺点：内存释放可能不够彻底
   - 适用：需要频繁进行分布式通信的场景

2. CuMemAllocator内存管理 (args.colocate=True)
   - 优点：内存释放更彻底，支持精确控制
   - 缺点：可能与PyTorch分布式通信冲突
   - 适用：内存受限且不需要频繁分布式通信的场景

3. PyTorch兼容内存池管理 (args.colocate=False, args.use_pytorch_pool=True)
   - 优点：与PyTorch完全兼容，支持内存池管理
   - 缺点：需要手动管理内存池
   - 适用：需要内存池管理但又要保持PyTorch兼容性

配置示例：

# 方案1：标准PyTorch内存管理
args.colocate = False
args.use_pytorch_pool = False

# 方案2：CuMemAllocator内存管理
args.colocate = True
args.use_pytorch_pool = False

# 方案3：PyTorch兼容内存池管理
args.colocate = False
args.use_pytorch_pool = True

使用PyTorch内存池的步骤：
1. 设置args.use_pytorch_pool = True
2. 在需要管理内存的张量上调用register_tensor_with_pool()
3. 调用sleep()和wake_up()进行内存管理

配置建议：
- 如果遇到SIGSEGV错误，请设置args.colocate=False
- 如果内存不足且不需要频繁通信，可以尝试args.colocate=True
- 多任务训练建议使用PyTorch兼容内存池
- 需要内存池管理时，使用PyTorch兼容内存池
"""

import ray
import torch
import wandb
import asyncio
import torch.distributed as dist

if torch.version.hip:
    from vllm.device_allocator.cumem import CuMemAllocator
else:
    from cumem_allocator import CuMemAllocator

from megatron.core import mpu

from transformers import AutoConfig, AutoTokenizer
from tracer import vinit, TracePoint, MemTracePoint

from slime.ray.ppo_actor import TrainRayActor
from slime.utils.memory_utils import clear_memory, print_memory
from slime.utils.timer import Timer, timer
from slime.utils.misc import ActorStatus
from slime.utils.wandb_utils import init_wandb_common

from ..utils.data import process_rollout_data
from .checkpoint import load_checkpoint
from .data import get_data_iterator, log_eval_data, log_perf_data, log_rollout_data
from .initialize import get_gloo_group, init, is_megatron_main_rank
from .loss import compute_advantages_and_returns
from .model import forward_only, initialize_model_and_optimizer, save, train
from .update_weight_utils import (
    named_parameters,
    UpdateWeightFromTensor,
    UpdateWeightFromDistributed,
)


class PyTorchMemoryPoolManager:
    """PyTorch兼容的内存池管理器"""
    
    def __init__(self, task_id):
        self.task_id = task_id
        self.memory_pool = {}  # 存储分配的内存块
        self.cpu_storage = {}  # CPU上的数据副本
        self.pool_tag = f"pytorch_pool_{task_id}"
        
    def allocate_tensor(self, size, dtype, device, name):
        """分配张量到内存池"""
        if device.type == 'cuda':
            # 在GPU上分配内存
            tensor = torch.empty(size, dtype=dtype, device=device)
            self.memory_pool[name] = {
                'tensor': tensor,
                'size': size,
                'dtype': dtype,
                'device': device,
                'allocated': True
            }
            return tensor
        else:
            # CPU张量直接分配
            return torch.empty(size, dtype=dtype, device=device)
    
    def deallocate_tensor(self, name):
        """释放内存池中的张量"""
        if name in self.memory_pool:
            # 将数据移动到CPU保存
            tensor = self.memory_pool[name]['tensor']
            if tensor.is_cuda:
                # 保存到CPU
                self.cpu_storage[name] = tensor.detach().cpu()
                
                # 强制删除GPU张量
                del tensor
                
                # 立即清理引用
                self.memory_pool[name]['allocated'] = False
                self.memory_pool[name]['tensor'] = None
                
                # 强制垃圾回收这个张量
                import gc
                gc.collect()
    
    def reallocate_tensor(self, name):
        """重新分配内存池中的张量"""
        if name in self.memory_pool and not self.memory_pool[name]['allocated']:
            info = self.memory_pool[name]
            # 重新分配GPU内存
            tensor = torch.empty(info['size'], dtype=info['dtype'], device=info['device'])
            # 从CPU恢复数据
            if name in self.cpu_storage:
                tensor.copy_(self.cpu_storage[name])
            self.memory_pool[name]['tensor'] = tensor
            self.memory_pool[name]['allocated'] = True
            return tensor
        return None
    
    def sleep(self):
        """将内存池中的所有张量移动到CPU"""
        print(f"PyTorchMemoryPool: Moving all tensors to CPU for task {self.task_id}")
        
        # 记录释放前的内存使用情况
        before_allocated = sum(1 for info in self.memory_pool.values() if info['allocated'])
        before_memory = torch.cuda.memory_allocated() / (1024**3)
        before_reserved = torch.cuda.memory_reserved() / (1024**3)
        
        print(f"PyTorchMemoryPool: Before sleep - {before_allocated} tensors allocated, {before_memory:.2f} GB allocated, {before_reserved:.2f} GB reserved")
        
        # 第一步：将所有张量移动到CPU
        for name in list(self.memory_pool.keys()):
            if self.memory_pool[name]['allocated']:
                print(f"PyTorchMemoryPool: Moving tensor '{name}' to CPU")
                self.deallocate_tensor(name)
        
        # 第二步：强制垃圾回收
        import gc
        gc.collect()
        
        # 第三步：清空PyTorch缓存
        torch.cuda.empty_cache()
        
        # 第四步：更激进的内存释放 - 重置PyTorch内存分配器
        if hasattr(torch.cuda, 'memory_stats'):
            # 尝试释放所有未使用的内存
            torch.cuda.memory.empty_cache()
            
            # 如果支持，尝试重置内存分配器
            if hasattr(torch.cuda, 'reset_peak_memory_stats'):
                torch.cuda.reset_peak_memory_stats()
        
        # 第五步：再次强制垃圾回收
        gc.collect()
        torch.cuda.empty_cache()
        
        # 记录释放后的内存使用情况
        after_allocated = sum(1 for info in self.memory_pool.values() if info['allocated'])
        after_memory = torch.cuda.memory_allocated() / (1024**3)
        after_reserved = torch.cuda.memory_reserved() / (1024**3)
        
        print(f"PyTorchMemoryPool: After sleep - {after_allocated} tensors allocated, {after_memory:.2f} GB allocated, {after_reserved:.2f} GB reserved")
        print(f"PyTorchMemoryPool: Freed {before_memory - after_memory:.2f} GB allocated memory, {before_reserved - after_reserved:.2f} GB reserved memory")
        
        # 如果内存释放效果不好，尝试更激进的方法
        if before_memory - after_memory < 0.5:  # 如果释放少于500MB
            print(f"PyTorchMemoryPool: Warning - Limited memory freed, trying more aggressive approach")
            self._aggressive_memory_cleanup()
    
    def _aggressive_memory_cleanup(self):
        """激进的内存清理方法"""
        print(f"PyTorchMemoryPool: Starting aggressive memory cleanup")
        
        # 记录清理前的内存
        before_memory = torch.cuda.memory_allocated() / (1024**3)
        before_reserved = torch.cuda.memory_reserved() / (1024**3)
        
        # 方法1：强制释放所有未使用的内存
        torch.cuda.empty_cache()
        
        # 方法2：尝试释放PyTorch的内存池
        if hasattr(torch.cuda, 'memory_stats'):
            try:
                # 获取内存统计信息
                stats = torch.cuda.memory_stats()
                print(f"PyTorchMemoryPool: Memory stats before cleanup: {stats}")
                
                # 尝试释放缓存的内存
                torch.cuda.memory.empty_cache()
            except Exception as e:
                print(f"PyTorchMemoryPool: Error in memory stats: {e}")
        
        # 方法3：多次垃圾回收
        import gc
        for i in range(3):
            gc.collect()
            torch.cuda.empty_cache()
        
        # 方法4：尝试重置内存分配器（如果支持）
        try:
            if hasattr(torch.cuda, 'reset_peak_memory_stats'):
                torch.cuda.reset_peak_memory_stats()
        except Exception as e:
            print(f"PyTorchMemoryPool: Error resetting peak memory stats: {e}")
        
        # 记录清理后的内存
        after_memory = torch.cuda.memory_allocated() / (1024**3)
        after_reserved = torch.cuda.memory_reserved() / (1024**3)
        
        print(f"PyTorchMemoryPool: Aggressive cleanup freed {before_memory - after_memory:.2f} GB allocated, {before_reserved - after_reserved:.2f} GB reserved")
    
    def wake_up(self):
        """将CPU上的数据恢复到GPU"""
        print(f"PyTorchMemoryPool: Restoring tensors to GPU for task {self.task_id}")
        
        # 记录恢复前的内存使用情况
        before_allocated = sum(1 for info in self.memory_pool.values() if info['allocated'])
        before_memory = torch.cuda.memory_allocated() / (1024**3)
        
        print(f"PyTorchMemoryPool: Before wake_up - {before_allocated} tensors allocated, {before_memory:.2f} GB GPU memory")
        
        restored_count = 0
        for name in list(self.memory_pool.keys()):
            if not self.memory_pool[name]['allocated']:
                print(f"PyTorchMemoryPool: Restoring tensor '{name}'")
                self.reallocate_tensor(name)
                restored_count += 1
        
        # 记录恢复后的内存使用情况
        after_allocated = sum(1 for info in self.memory_pool.values() if info['allocated'])
        after_memory = torch.cuda.memory_allocated() / (1024**3)
        
        print(f"PyTorchMemoryPool: After wake_up - {after_allocated} tensors allocated, {after_memory:.2f} GB GPU memory")
        print(f"PyTorchMemoryPool: Restored {restored_count} tensors, allocated {after_memory - before_memory:.2f} GB GPU memory")
    
    def get_memory_usage(self):
        """获取内存使用信息"""
        allocated_count = sum(1 for info in self.memory_pool.values() if info['allocated'])
        total_count = len(self.memory_pool)
        cpu_storage_size = sum(tensor.numel() * tensor.element_size() 
                              for tensor in self.cpu_storage.values())
        
        return {
            'allocated_tensors': allocated_count,
            'total_tensors': total_count,
            'cpu_storage_mb': cpu_storage_size / (1024 * 1024),
            'pool_tag': self.pool_tag
        }
    
    def get_detailed_memory_info(self):
        """获取详细的内存信息"""
        info = self.get_memory_usage()
        
        # 计算GPU内存使用
        total_gpu_memory = 0
        for name, pool_info in self.memory_pool.items():
            if pool_info['allocated'] and pool_info['tensor'] is not None:
                tensor = pool_info['tensor']
                if tensor.is_cuda:
                    total_gpu_memory += tensor.numel() * tensor.element_size()
        
        info['gpu_memory_mb'] = total_gpu_memory / (1024 * 1024)
        info['tensor_names'] = list(self.memory_pool.keys())
        
        return info


class MegatronTrainRayActor(TrainRayActor):
    def init(self, args, role, with_ref=False):
        tp = TracePoint(f"task-{args.task_id}: Megatron train actor init", "1")
        tp.begin()
        super().init(args, role, with_ref)
        
        if self._rank == 0:
            wandb.init(
                project=args.wandb_project+str(args.task_id),
                group=f"{args.wandb_group}-{args.task_id}",
                name=f"{args.task_id}-MegatronTrainRayActor-{self._rank}",
                config={"rank": self._rank},
            )
            init_wandb_common()

        init(args)

        # read config and tokenizer serialized to prevent concurrent writing bug.
        for i in range(dist.get_world_size()):
            if i == dist.get_rank():
                c_tp = TracePoint(f"task-{self.args.task_id}: read config and tokenizer", "1")
                c_tp.begin()
                self.hf_config = AutoConfig.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
                self.tokenizer = AutoTokenizer.from_pretrained(self.args.hf_checkpoint, trust_remote_code=True)
                c_tp.end()
            dist.barrier(group=get_gloo_group(self.args.task_id))

        if self.args.debug_rollout_only:
            Timer().start("train_wait")
            return 0

        m_op = TracePoint(f"task-{self.args.task_id}: init model and optimizer", "1")
        m_op.begin()
        MemTracePoint.record("before init model and optimizer")
        (self.model, self.optimizer, self.opt_param_scheduler, loaded_rollout_id) = initialize_model_and_optimizer(
            args
        )
        MemTracePoint.record("after init model and optimizer")
        m_op.end()

        lp = TracePoint(f"task-{self.args.task_id}: update model in cpu", "1")
        lp.begin()
        MemTracePoint.record("before update model in cpu")
        start_rollout_id = loaded_rollout_id + 1
        
        # 初始化内存管理器
        if getattr(self.args, 'colocate', False):
            # 使用CuMemAllocator
            self.memory_manager = MemoryPoolContentManager(args, self.model, self.optimizer)
            self.pytorch_pool_manager = None
        else:
            # 使用PyTorch兼容内存池
            self.memory_manager = None
            self.pytorch_pool_manager = PyTorchMemoryPoolManager(self.args.task_id)
        
        # 保持向后兼容的weights结构
        self.weights = {"actor": {}}
        self.update_cpu_params_dict(self.weights["actor"])
        
        # 初始化内存池内容到CPU（如果使用内存管理器）
        if self.memory_manager:
            self.memory_manager.update_cpu_all()
        elif self.pytorch_pool_manager:
            # 自动注册模型参数到PyTorch内存池
            print(f"Task {self.args.task_id}: Registering model parameters to PyTorch memory pool")
            # self.model是一个列表，需要遍历每个模型块
            for model_chunk_idx, model_chunk in enumerate(self.model):
                for name, param in model_chunk.named_parameters():
                    if param.is_cuda:
                        self.register_tensor_with_pool(param, f"model_param_chunk{model_chunk_idx}_{name}")
            
            # 自动注册优化器状态到PyTorch内存池
            print(f"Task {self.args.task_id}: Registering optimizer states to PyTorch memory pool")
            optimizer_state = self.optimizer.state_dict()
            
            print(f"Task {self.args.task_id}: Optimizer state_dict keys: {list(optimizer_state.keys())}")
            
            # 安全地访问优化器状态
            if 'state' in optimizer_state:
                state_count = 0
                for param_name, param_state in optimizer_state['state'].items():
                    for state_key, state_value in param_state.items():
                        if isinstance(state_value, torch.Tensor) and state_value.is_cuda:
                            self.register_tensor_with_pool(state_value, f"optim_{param_name}_{state_key}")
                            state_count += 1
                print(f"Task {self.args.task_id}: Registered {state_count} optimizer state tensors")
            else:
                print(f"Task {self.args.task_id}: Warning - optimizer state_dict does not contain 'state' key")
                print(f"Task {self.args.task_id}: Available keys in optimizer state_dict: {list(optimizer_state.keys())}")
                
                # 尝试注册其他可能的优化器状态
                for key, value in optimizer_state.items():
                    if isinstance(value, torch.Tensor) and value.is_cuda:
                        self.register_tensor_with_pool(value, f"optim_{key}")
                        print(f"Task {self.args.task_id}: Registered optimizer tensor '{key}'")
                    elif isinstance(value, dict):
                        for sub_key, sub_value in value.items():
                            if isinstance(sub_value, torch.Tensor) and sub_value.is_cuda:
                                self.register_tensor_with_pool(sub_value, f"optim_{key}_{sub_key}")
                                print(f"Task {self.args.task_id}: Registered optimizer tensor '{key}_{sub_key}'")
            
            # 打印内存池状态
            memory_info = self.pytorch_pool_manager.get_memory_usage()
            print(f"Task {self.args.task_id}: PyTorch memory pool initialized with {memory_info}")
            
            # 打印详细的内存信息
            detailed_info = self.pytorch_pool_manager.get_detailed_memory_info()
            print(f"Task {self.args.task_id}: Detailed memory info: {detailed_info['gpu_memory_mb']:.2f} MB GPU memory in pool")
            
            # 显示前几个张量的信息作为示例
            if detailed_info['tensor_names']:
                print(f"Task {self.args.task_id}: Sample tensors in pool:")
                for i, name in enumerate(detailed_info['tensor_names'][:5]):
                    pool_info = self.pytorch_pool_manager.memory_pool[name]
                    if pool_info['allocated'] and pool_info['tensor'] is not None:
                        tensor = pool_info['tensor']
                        size_mb = tensor.numel() * tensor.element_size() / (1024 * 1024)
                        print(f"  {i+1}. {name}: {tensor.shape}, {tensor.dtype}, {size_mb:.2f} MB")
                if len(detailed_info['tensor_names']) > 5:
                    print(f"  ... and {len(detailed_info['tensor_names']) - 5} more tensors")
        
        MemTracePoint.record("after update model in cpu")
        lp.end()

        locp = TracePoint(f"task-{self.args.task_id}: load other model", "1")
        locp.begin()
        if with_ref:
            MemTracePoint.record("before load ref model")
            self.load_other_checkpoint("ref", args.ref_load)
            MemTracePoint.record("after load ref model")

        if self.args.keep_old_actor:
            MemTracePoint.record("before load old actor model")
            self.load_other_checkpoint("old_actor", args.load)
            MemTracePoint.record("after load old actor model")

        locp.end()

        if self.args.offload:
            offp = TracePoint(f"task-{self.args.task_id}: offload model to cpu", "1")
            offp.begin()
            MemTracePoint.record("before update model in gpu")
            # recover to actor in the end.
            self.update_gpu_params_dict(self.weights["actor"])
            MemTracePoint.record("after update model in gpu")
            MemTracePoint.record("before offload model to cpu")
            self.sleep(("model"))
            MemTracePoint.record("after offload model to cpu")
            offp.end()

        uwp = TracePoint(f"task-{self.args.task_id}: init update weight cls", "1")
        uwp.begin()
        update_weight_cls = UpdateWeightFromTensor if self.args.colocate else UpdateWeightFromDistributed
        self.weight_updator = update_weight_cls(
            self.args,
            self.model,
            self.weights,
            model_name=type(self.hf_config).__name__.lower() if self.args.model_name is None else self.args.model_name,
            quantization_config=getattr(self.hf_config, "quantization_config", None),
            vocab_size=self.tokenizer.vocab_size if self.args.vocab_size is None else self.args.vocab_size,
        )
        uwp.end()

        MemTracePoint.record("before empty cache")
        # empty cache after initialization
        clear_memory()
        MemTracePoint.record("after empty cache")

        self.rollout_engines = None
        self.data_buffer = None

        self.rollout_data_postprocess = None
        if self.args.rollout_data_postprocess_path is not None:
            from slime.utils.misc import load_function

            self.rollout_data_postprocess = load_function(self.args.rollout_data_postprocess_path)

        Timer().start("train_wait")
        
        # 检查内存池初始化状态
        if hasattr(self, 'pytorch_pool_manager') and self.pytorch_pool_manager:
            self.check_memory_pool_initialization()
        
        tp.end()
        return start_rollout_id

    def update_cpu_params_dict(self, params_dict):
        with torch.no_grad():
            for name, param in named_parameters(self.args, self.model):
                if name not in params_dict:
                    params_dict[name] = torch.empty_like(param, device=torch.device("cpu"), pin_memory=True)
                params_dict[name].copy_(param.detach(), non_blocking=True)
            torch.cuda.synchronize()

    def update_gpu_params_dict(self, params_dict):
        with torch.no_grad():
            for name, param in named_parameters(self.args, self.model):
                assert name in params_dict
                param.copy_(params_dict[name], non_blocking=True)
            torch.cuda.synchronize()

    def sleep(self, tags):
        with timer("sleep"):
            tp = TracePoint(f"task-{self.args.task_id}: actor model real sleep", "1")
            tp.begin()
            MemTracePoint.record("before offload actor model")
            assert self.status == ActorStatus.ONLOAD or self.status == ActorStatus.PENDING
            assert self.args.offload
            tags = tags+str(self.args.task_id)
            assert "model"+str(self.args.task_id) in tags
            if isinstance(tags, str):
                tags = (tags,)

            clear_memory()
            print_memory(f"before offload actor model")
            self.update_cpu_params_dict(self.weights["actor"])

            # 根据不同的内存管理模式执行卸载
            if getattr(self.args, 'colocate', False) and hasattr(self, 'memory_manager') and self.memory_manager:
                # 使用CuMemAllocator释放内存池
                allocator = CuMemAllocator.get_instance()
                allocator.sleep(offload_tags=tags)
                print(f"Task {self.args.task_id}: Released CuMemAllocator memory pool")
                
            else:
                # 使用PyTorch兼容内存池
                print(f"Task {self.args.task_id}: Using PyTorch memory pool")
                memory_info = self.pytorch_pool_manager.get_memory_usage()
                print(f"Task {self.args.task_id}: Before sleep - PyTorch memory pool info: {memory_info}")
                
                self.pytorch_pool_manager.sleep()
                
                memory_info_after = self.pytorch_pool_manager.get_memory_usage()
                print(f"Task {self.args.task_id}: After sleep - PyTorch memory pool info: {memory_info_after}")
                

            clear_memory()
            print_memory(f"after offload actor model")
            self.status = ActorStatus.OFFLOAD
            MemTracePoint.record("after onload actor model")
            tp.end()

    def wake_up(self, tags):
        with timer("wake up"):
            tp = TracePoint(f"task-{self.args.task_id}: actor model real wake up", "1")
            tp.begin()
            MemTracePoint.record("before onload actor model")
            assert self.status == ActorStatus.OFFLOAD
            assert self.args.offload
            tags = tags+str(self.args.task_id) 
            clear_memory()
            print_memory("before wake_up actor model")

            if isinstance(tags, str):
                tags = (tags,)

            # 根据不同的内存管理模式执行恢复
            if getattr(self.args, 'colocate', False) and hasattr(self, 'memory_manager') and self.memory_manager:
                # 重新分配CuMemAllocator内存池
                allocator = CuMemAllocator.get_instance()
                allocator.wake_up(tags)
                print(f"Task {self.args.task_id}: Reallocated CuMemAllocator memory pool")
                
            else:
                # 恢复PyTorch兼容内存池
                self.pytorch_pool_manager.wake_up()
                print(f"Task {self.args.task_id}: Restored PyTorch memory pool")

            clear_memory()
            print_memory("after wake_up actor model")
            MemTracePoint.record("after offload actor model")
            self.status = ActorStatus.ONLOAD
            tp.end()

    async def set_data_buffer(self, data_buffer):
        tp = TracePoint(f"task-{self.args.task_id}: megatron train actor set data buffer", "1")
        tp.begin()
        self.data_buffer = data_buffer
        tp.end()

    async def get_rollout_data(self, rollout_id, rollout_data):
        # Fetch data through ray on CPU, not sure if this will be performance bottleneck.
        # Both first pp stage and the last pp stage will recieve the data.
        await process_rollout_data(
            rollout_id,
            self.args,
            self.data_buffer,
            mpu.get_data_parallel_rank(with_context_parallel=False),
            mpu.get_data_parallel_world_size(with_context_parallel=False),
            rollout_data=rollout_data,
        )

    def compute_log_prob(
        self,
        model_tag,
        log_probs_data_iterator,
        log_probs_num_microbatches,
        store_prefix="",
        rollout_data=None,
    ):
        # reset data iterator
        for data_iterator in log_probs_data_iterator:
            data_iterator.reset()

        self.update_gpu_params_dict(self.weights[model_tag])

        with timer(f"{store_prefix}log_probs"):
            forward_only(
                self.args,
                self.model,
                log_probs_data_iterator,
                log_probs_num_microbatches,
                store_prefix=store_prefix,
                rollout_data=rollout_data,
            )

    async def train(self, rollout_id, with_data_fetching=True):
        if self.args.offload:
            assert self.status == ActorStatus.ONLOAD
        # if self._task_id == 1:
        #     breakpoint()
        Timer().end("train_wait")
        prepare_train = TracePoint(f"task-{self.args.task_id}: megatron train actor prepare train", "1")
        prepare_train.begin()

        rollout_data = {}

        if self.args.debug_rollout_only:
            # For debug rollout, we just log the data and return.
            if with_data_fetching:
                data_fetch_trace = TracePoint(f"task-{self.args.task_id}: megatron train actor data fetch", "1")
                data_fetch_trace.begin()
                await self.get_rollout_data(rollout_id, rollout_data)
                data_fetch_trace.end()

            log_rollout_data(rollout_id, self.args, rollout_data)
            log_perf_data(rollout_id, self.args)
            train_trace.end()
            Timer().start("train_wait")
            return

        with timer("train"):
            with timer("data_preprocess"):
                # For async train, we need to separate the data fetching and training.
                if with_data_fetching:
                    data_fetch_trace = TracePoint(f"task-{self.args.task_id}: megatron train actor data fetch", "1")
                    data_fetch_trace.begin()
                    await self.get_rollout_data(rollout_id, rollout_data)
                    data_fetch_trace.end()
                    MemTracePoint.record("megatron train actor after data fetch")

                # Create data iterator for log_probs and train.
                (
                    log_probs_data_iterator,
                    log_probs_num_microbatches,
                    train_data_iterator,
                    train_num_microbatches,
                ) = get_data_iterator(self.args, self.model, rollout_data)

            if self.args.compute_advantages_and_returns:
                if "ref" in self.weights:
                    ref_trace = TracePoint(f"task-{self.args.task_id}: megatron train actor ref log prob", "1")
                    ref_trace.begin()
                    ref_update = TracePoint(f"task-{self.args.task_id}: megatron train actor update ref in gpu", "1")
                    ref_update.begin()
                    MemTracePoint.record("before update ref in gpu")
                    self.update_gpu_params_dict(self.weights["ref"])
                    MemTracePoint.record("after update ref in gpu")
                    ref_update.end()
                    self.compute_log_prob(
                        "ref",
                        log_probs_data_iterator,
                        log_probs_num_microbatches,
                        store_prefix="ref_",
                        rollout_data=rollout_data,
                    )
                    ref_trace.end()

                actor_trace = TracePoint(f"task-{self.args.task_id}: megatron train actor actor log prob", "1")
                actor_trace.begin()
                self.compute_log_prob(
                    "old_actor" if self.args.keep_old_actor else "actor",
                    log_probs_data_iterator,
                    log_probs_num_microbatches,
                    store_prefix="",
                    rollout_data=rollout_data,
                )
                actor_trace.end()
                # when there is old actor, we need to update the model params to actor manually
                if "old_actor" in self.weights:
                    MemTracePoint.record("before update old actor in gpu")
                    self.update_gpu_params_dict(self.weights["actor"])
                    MemTracePoint.record("after update old actor in gpu")

                # Calculate adv and returns. Need to performed before training (instead of on the fly),
                # because we may need normalize the whole rollout.
                compute_advantages_and_returns(self.args, rollout_data)

            if self.rollout_data_postprocess is not None:
                postprocess_trace = TracePoint(f"task-{self.args.task_id}: megatron train actor data post process", "1")
                postprocess_trace.begin()
                self.rollout_data_postprocess(self.args)
                postprocess_trace.end()

            log_trace = TracePoint(f"task-{self.args.task_id}: megatron train actor log rollout", "1")
            log_trace.begin()
            log_rollout_data(rollout_id, self.args, rollout_data)
            log_trace.end()

            prepare_train.end()
            # Train
            with timer("actor_train"):
                tp = TracePoint(f"task-{self.args.task_id}: megatron train actor real train", "1")
                tp.begin()
                train(
                    rollout_id,
                    self.args.wandb_run_id,
                    self.model,
                    self.optimizer,
                    self.opt_param_scheduler,
                    train_data_iterator,
                    train_num_microbatches,
                )
                tp.end()

        log_perf_data(rollout_id, self.args)
        Timer().start("train_wait")

    async def eval(self, rollout_id):
        if self.args.debug_train_only:
            return

        # TODO: is logging enough?
        await log_eval_data(rollout_id, self.args, self.data_buffer)

    async def save_model(self, iteration, with_optimizer=True):
        tp = TracePoint(f"task-{self.args.task_id}: save model", "1")
        tp.begin()
        MemTracePoint.record("before save model")

        if self.args.debug_rollout_only:
            tp.end()
            return

        if with_optimizer:
            save(iteration, self.model, self.optimizer, self.opt_param_scheduler)
        else:
            save(iteration, self.model, None, None)

        MemTracePoint.record("after save model")
        tp.end()

    async def connect_rollout_engines(self, rollout_engines, rollout_engine_lock):
        tp = TracePoint(f"task-{self.args.task_id}: connect rollout engines", "1")
        tp.begin()
        
        self.rollout_engines = rollout_engines

        if self.args.debug_train_only or self.args.debug_rollout_only:
            tp.end()
            return

        MemTracePoint.record("before connect weight updator")
        await self.weight_updator.connect_rollout_engines(rollout_engines, rollout_engine_lock)
        MemTracePoint.record("after connect weight updator")

        dist.barrier(group=get_gloo_group(self.args.task_id))
        tp.end()

    async def update_weights(self):
        tp = TracePoint(f"task-{self.args.task_id}: update weights", "1")
        tp.begin()
        MemTracePoint.record("before update weights")

        with timer("update_weight"):
            if self.args.debug_train_only or self.args.debug_rollout_only:
                return

            MemTracePoint.record("before empty cache")
            torch.cuda.empty_cache()
            MemTracePoint.record("after empty cache")

            MemTracePoint.record("before weight update")
            await self.weight_updator.update_weights()
            MemTracePoint.record("after weight update")

            dist.barrier(group=get_gloo_group(self.args.task_id))
            
            MemTracePoint.record("before clear memory")
            clear_memory()
            MemTracePoint.record("after clear memory")
            print_memory("after update_weights")

            if getattr(self.args, "keep_old_actor", False):
                MemTracePoint.record("before update old actor")
                print("update rollout model on cpu using actor model")
                self.update_cpu_params_dict(self.weights["old_actor"])
                MemTracePoint.record("after update old actor")

        tp.end()

    def load_other_checkpoint(self, model_tag, path):
        old_args = self.args.load, self.args.no_load_optim, self.args.no_load_rng, self.args.finetune
        self.args.load = path
        self.args.no_load_optim = True
        self.args.no_load_rng = True
        self.args.finetune = True
        _, _ = load_checkpoint(
            self.model,
            None,
            None,
            checkpointing_context={},
            skip_load_to_model_and_opt=False,
        )
        self.args.load, self.args.no_load_optim, self.args.no_load_rng, self.args.finetune = old_args

        self.weights[model_tag] = {}
        self.update_cpu_params_dict(self.weights[model_tag])

    def register_tensor_with_pool(self, tensor, name):
        """将张量注册到PyTorch内存池中"""
        if hasattr(self, 'pytorch_pool_manager') and self.pytorch_pool_manager and tensor.is_cuda:
            # 将张量信息注册到内存池
            self.pytorch_pool_manager.memory_pool[name] = {
                'tensor': tensor,
                'size': tensor.size(),
                'dtype': tensor.dtype,
                'device': tensor.device,
                'allocated': True
            }
            print(f"Registered tensor '{name}' with PyTorch memory pool")
    
    def get_memory_pool_status(self):
        """获取内存池状态信息"""
        if hasattr(self, 'pytorch_pool_manager') and self.pytorch_pool_manager:
            return self.pytorch_pool_manager.get_memory_usage()
        elif hasattr(self, 'memory_manager') and self.memory_manager:
            return self.memory_manager.get_memory_usage_info()
        else:
            return {
                'mode': 'standard_pytorch',
                'message': 'Using standard PyTorch memory management'
            }
    
    def check_memory_pool_initialization(self):
        """检查内存池是否正确初始化"""
        if hasattr(self, 'pytorch_pool_manager') and self.pytorch_pool_manager:
            memory_info = self.pytorch_pool_manager.get_memory_usage()
            print(f"Task {self.args.task_id}: Memory pool check:")
            print(f"  - Total tensors in pool: {memory_info['total_tensors']}")
            print(f"  - Allocated tensors: {memory_info['allocated_tensors']}")
            print(f"  - CPU storage size: {memory_info['cpu_storage_mb']:.2f} MB")
            
            if memory_info['total_tensors'] == 0:
                print(f"  - WARNING: Memory pool is empty! No tensors registered.")
                return False
            else:
                print(f"  - SUCCESS: Memory pool initialized with {memory_info['total_tensors']} tensors")
                return True
        else:
            print(f"Task {self.args.task_id}: No PyTorch memory pool manager found")
            return False
