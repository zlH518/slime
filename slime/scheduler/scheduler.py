import ray
import asyncio

from .task import Task
from tracer import vinit, TracePoint

class Scheduler:
    """
    A Scheduler for multi task running in parallelism
    """
    def __init__(self, args):
        self.tasks_args = args.tasks_args
        self.tasks_num = args.tasks_num
        
        assert len(self.tasks_args) == self.tasks_num

        self.tasks = []

        self.train_lock= asyncio.Semaphore(args.train_semaphore)
        self.rollout_lock = asyncio.Semaphore(args.rollout_semaphore)
        self.update_weight_lock = asyncio.Semaphore(args.update_weight_semaphore)

        self.pgs = args.pgs

        self.tasks = [Task(args) for args in self.tasks_args]
        vinit()


    async def init_task(self):
        print("Initializing tasks...")
        for task in self.tasks:
            tp = TracePoint(f"task-{task.args.task_id}: init", "1")
            tp.begin()
            await task.init()
            tp.end()
        print("All tasks initialized.")


    async def run(self):
        print("Starting concurrent training tasks...")

        await asyncio.gather(*[self.train(task_id) for task_id in range(self.tasks_num)])

        print("All concurrent training tasks finished.")


    async def train(self, task_id):
        args = self.tasks[task_id].args
        for rollout_id in range(args.start_rollout_id, args.num_rollout):
            async with self.rollout_lock:
                # if args.offload:
                #     tp = TracePoint(f"task-{task_id}: rollout model onload", "1")
                #     tp.begin()
                #     await asyncio.gather(*(self.tasks[task_id].rollout_generator.async_onload()))
                #     tp.end()
            
                tp = TracePoint(f"task-{task_id}: rollout", "1")
                tp.begin()
                await self.tasks[task_id].rollout_generator.async_generate(rollout_id)
                tp.end()

                if args.offload:
                    tp = TracePoint(f"task-{task_id}: rollout model offload", "1")
                    tp.begin()
                    await asyncio.gather(*(self.tasks[task_id].rollout_generator.async_offload()))
                    tp.end()

            async with self.train_lock:
                tp = TracePoint(f"task-{task_id}: train", "1")
                tp.begin()
                await asyncio.gather(*self.tasks[task_id].actor_model.async_train(rollout_id))
                tp.end()

                if args.save_interval is not None and (
                    (rollout_id + 1) % args.save_interval == 0
                    or (self.tasks[task_id].num_rollout_per_epoch is not None and (rollout_id + 1) % self.tasks[task_id].num_rollout_per_epoch == 0)
                ):
                    tp = TracePoint(f"task-{task_id}: save actor model", "1")
                    tp.begin()
                    await asyncio.gather(*(self.tasks[task_id].actor_model.async_save_model(rollout_id)))
                    tp.end()
                    
                    if args.rollout_global_dataset:
                        tp = TracePoint(f"task-{task_id}: data buffer save data", "1")
                        tp.begin()
                        await self.tasks[task_id].rollout_generator.data_buffer.save.remote(rollout_id)
                        tp.end()

                if args.offload:
                    tp = TracePoint(f"task-{task_id}: actor model offload", "1")
                    tp.begin()
                    await asyncio.gather(*(self.tasks[task_id].actor_model.async_offload()))
                    tp.end()

                    tp = TracePoint(f"task-{task_id}: rollout model onload", "1")
                    tp.begin()
                    await asyncio.gather(*(self.tasks[task_id].rollout_generator.async_onload()))
                    tp.end()


            async with self.update_weight_lock:
                tp = TracePoint(f"task-{task_id}: update weight", "1")
                tp.begin()
                await asyncio.gather(*(self.tasks[task_id].actor_model.async_update_weights()))
                tp.end()

                # if args.offload:
                #     tp = TracePoint(f"task-{task_id}: rollout model offload", "1")
                #     tp.begin()
                #     await asyncio.gather(*(self.tasks[task_id].rollout_generator.async_offload()))
                #     tp.end()