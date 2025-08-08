import ray
import wandb
import asyncio

from .task import Task
from tracer import vinit, TracePoint
from slime.utils.timer import Timer, timer

class Scheduler:
    """
    A Scheduler for multi task running in parallelism
    """
    def __init__(self, args):
        self.tasks_args = args.tasks_args
        self.tasks_num = args.tasks_num
        
        assert len(self.tasks_args) == self.tasks_num

        self.tasks = []

        self.train_lock= asyncio.Lock()
        self.rollout_lock = asyncio.Lock()


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
        print("9"*200)
        print("All tasks initialized.")


    async def run(self):
        print("Starting concurrent training tasks...")

        await asyncio.gather(*[self.train(task_id) for task_id in range(self.tasks_num)])

        print("All concurrent training tasks finished.")
        print("6"*200)
        for task_id in range(self.tasks_num):
            print(f"task-{task_id} end to end time: ",end="")
            print(Timer().log_dict()[f"task-{task_id}"]) 
        print("6"*200)


    async def train(self, task_id):
        # breakpoint()
        Timer().start(f"task-{task_id}")
        args = self.tasks[task_id].args
        for rollout_id in range(args.start_rollout_id, args.num_rollout):
            async with self.rollout_lock:
                if args.offload:
                    rao_p = TracePoint(f"task-{task_id}: rollout and actor model onload", "1")
                    rao_p.begin()
                    r_and_a_on_ref = []
                    r_and_a_on_ref.extend(self.tasks[task_id].rollout_generator.async_onload())
                    r_and_a_on_ref.extend(self.tasks[task_id].actor_model.async_onload())
                    await asyncio.gather(*r_and_a_on_ref)
                    rao_p.end()


                u_p = TracePoint(f"task-{task_id}: update weight", "1")
                u_p.begin()
                await asyncio.gather(*(self.tasks[task_id].actor_model.async_update_weights()))
                u_p.end()

                g_p = TracePoint(f"task-{task_id}: generate", "1")
                g_p.begin()
                generate = self.tasks[task_id].rollout_generator.async_generate(rollout_id)

                offload_ref = []
                if args.offload:
                    tp = TracePoint(f"task-{task_id}: actor and rollout model offload", "1")
                    tp.begin()
                    offload_ref.extend(self.tasks[task_id].actor_model.async_offload())

                    await generate
                    g_p.end()
                    offload_ref.extend(self.tasks[task_id].rollout_generator.async_offload())
                    tp.end()
                await asyncio.gather(*offload_ref)

            async with self.train_lock:
                if args.offload:
                    tp = TracePoint(f"task-{task_id}: actor model onload", "1")
                    tp.begin()
                    await asyncio.gather(*(self.tasks[task_id].actor_model.async_onload()))
                    tp.end()

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

        wandb.finish()
        Timer().end(f"task-{task_id}")