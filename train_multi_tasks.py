
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


    scheduler_args = SchedulerParams(
        tasks_num=0,
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