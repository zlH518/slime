import ray
import asyncio

from slime.utils.misc import Singleton

from .task import Task
from tracer import tracepoint_module_setup, TracePoint


class Scheduler(Singleton):
    """
    A Scheduler for multi task running in parallelism
    """
    def __init__(self, args):
        self.tasks_args = args.tasks_args
        self.tasks_num = len(args.tasks_num)
        
        assert len(tasks_args) == self.tasks_num

        self.tasks = []

        self.train_lock= asyncio.Semaphore(args.train_semaphore)
        self.rollout_lock = asyncio.Semaphore(args.rollout_semaphore)
        self.update_weight_lock = asyncio.Semaphore(args.update_weight_semaphore)

        self.pgs = args.pgs

        self.tasks = [Task(args) for args in self.tasks_args]


    async def init_task(self):
        print("Initializing tasks...")
        await asyncio.gather(*[task.init() for task in self.tasks])
        print("All tasks initialized.")


    async def run(self):
        print("Starting concurrent training tasks...")

        await asyncio.gather(*[self.train(task_id) for task_id in range(self.task_num)])

        print("All concurrent training tasks finished.")


    async def train(self,task_id):
        args = self.tasks[task_id].args
        for rollout_id in range(args.start_rollout_id, args.num_rollout):
            async with self.rollout_lock:
                await rollout_generator.async_generate(rollout_id)

                if args.offload:
                    await asyncio.gather(*(rollout_generator.async_offload()))

            async with self.train_lock:
                await asyncio.gather(*(actor_model.async_train(rollout_id)))

                if args.save_interval is not None and (
                    (rollout_id + 1) % args.save_interval == 0
                    or (num_rollout_per_epoch is not None and (rollout_id + 1) % num_rollout_per_epoch == 0)
                ):
                    await asyncio.gather(*(actor_model.async_save_model(rollout_id)))
                    if args.rollout_global_dataset:
                        await rollout_generator.data_buffer.save.remote(rollout_id)

                if args.offload:
                    ray.get(actor_model.async_offload())
                    ray.get(rollout_generator.async_onload())

            async with self.update_weight_lock:
                await asyncio.gather(*(actor_model.async_update_weights()))