
import os
import torch
import ray
import asyncio


from slime.scheduler import Scheduler, SchedulerParams
from slime.utils.arguments import parse_args


async def main(args):
    if not ray.is_initialized():
        print("--"*50)
        # ray.init(runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN", "RAY_DEBUG_POST_MORTEM": "1"}})
        ray.init(runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN"}})
    # print(f"Process CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
    # print(f"PyTorch device count: {torch.cuda.device_count()}")
    # print('ray cluster resources:', ray.cluster_resources())
    # print(ray.get_gpu_ids())
    # for i in range(torch.cuda.device_count()):
    #     print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    # TODO: use yaml to config scheduler
    scheduler_args = SchedulerParams(
        tasks_num=2,
        tasks_args_dir="xxx",
        tasks_args_template=args,
        train_semaphore=1,
        rollout_semaphore=1,
    )

    scheduler = Scheduler(scheduler_args)

    await scheduler.init_task()
    await scheduler.run()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args))
    print("finish")