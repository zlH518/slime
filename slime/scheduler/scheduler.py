"""
下面是分三个阶段做的并行任务，提高并行度
"""

import os
import ray
import time
import wandb
import asyncio

from .task import Task
from tracer import vinit, TracePoint
from slime.utils.timer import Timer, timer
from contextlib import redirect_stdout

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
        self.update_lock = asyncio.Lock()

        self.all_locks = [self.train_lock, self.rollout_lock]


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
        # breakpoint()
        if self.tasks_args[0].colocate:
            for task_id in range(self.tasks_num):
                await self.train(task_id)
        else:
            await asyncio.gather(*[self.train(task_id) for task_id in range(self.tasks_num)])

        print("All concurrent training tasks finished.")
        perf_path = os.getenv("PERF_DIR", "")
        assert perf_path is not ""
        log_path = os.path.join(perf_path, "perf.log")
        with open(log_path, "a") as f:
            with redirect_stdout(f):
                print("All concurrent training tasks finished.")
                print("6" * 200)
                for task_id in range(self.tasks_num):
                    print("*" * 200)
                    print(f"task-{task_id}")
                    for k, v in Timer().log_dict(task_id).items():
                        print(k, v)
                    print("*" * 200)
                print("6" * 200)


    async def train(self, task_id):
        # breakpoint()
        Timer().start(task_id, "task-runtime")
        args = self.tasks[task_id].args
        for rollout_id in range(args.start_rollout_id, args.num_rollout):
            async with self.update_lock:
                await asyncio.gather(*[lock.acquire() for lock in self.all_locks])
                start_time = time.time()
                if args.offload:
                    rao_p = TracePoint(f"task-{task_id}: rollout and actor model onload", "1")
                    rao_p.begin()
                    r_and_a_on_ref = []
                    r_and_a_on_ref.extend(self.tasks[task_id].rollout_generator.async_onload())
                    r_and_a_on_ref.extend(self.tasks[task_id].actor_model.async_onload())
                    await asyncio.gather(*r_and_a_on_ref)
                    rao_p.end()
                end_time = time.time()
                Timer().add(task_id, "task-preprocess-update", end_time-start_time)

                start_time = time.time()
                u_p = TracePoint(f"task-{task_id}: update weight", "1")
                u_p.begin()
                await asyncio.gather(*(self.tasks[task_id].actor_model.async_update_weights()))
                u_p.end()
                end_time = time.time()
                Timer().add(task_id, "task-update", end_time-start_time)

                start_time = time.time()
                if args.offload:
                    raoff_p = TracePoint(f"task-{task_id}: rollout and actor model offload", "1")
                    raoff_p.begin()
                    r_and_a_off_ref = []
                    r_and_a_off_ref.extend(self.tasks[task_id].rollout_generator.async_offload())
                    r_and_a_off_ref.extend(self.tasks[task_id].actor_model.async_offload())
                    await asyncio.gather(*r_and_a_off_ref)
                    raoff_p.end()
                for lock in self.all_locks:
                    lock.release()
                end_time = time.time()
                Timer().add(task_id, "task-postprocess-update", end_time-start_time)

            async with self.rollout_lock:
                start_time = time.time()
                if args.offload and not args.colocate:
                    ro_p = TracePoint(f"task-{task_id}: rollout model onload", "1")
                    ro_p.begin()
                    r_on_ref = []
                    r_on_ref.extend(self.tasks[task_id].rollout_generator.async_onload())
                    await asyncio.gather(*r_on_ref)
                    ro_p.end()
                end_time = time.time()
                Timer().add(task_id, "task-preprocess-rollout", end_time-start_time)

                start_time = time.time()
                g_p = TracePoint(f"task-{task_id}: generate", "1")
                g_p.begin()
                generate = self.tasks[task_id].rollout_generator.async_generate(rollout_id)
                await generate
                g_p.end()
                end_time = time.time()
                Timer().add(task_id, "task-rollout", end_time-start_time)

                start_time = time.time()
                rollout_offload_ref = []
                if args.offload:
                    roff_p = TracePoint(f"task-{task_id}: rollout model offload", "1")
                    roff_p.begin()
                    rollout_offload_ref.extend(self.tasks[task_id].rollout_generator.async_offload())
                    roff_p.end()
                await asyncio.gather(*rollout_offload_ref)
                end_time = time.time()
                Timer().add(task_id, "task-postprocess-rollout", end_time-start_time)

            async with self.train_lock:
                if args.offload:
                    start_time = time.time()
                    tp = TracePoint(f"task-{task_id}: actor model onload", "1")
                    tp.begin()
                    await asyncio.gather(*(self.tasks[task_id].actor_model.async_onload()))
                    tp.end()
                    end_time = time.time()
                    Timer().add(task_id, "task-preprocess-train", end_time-start_time)

                start_time = time.time()
                tp = TracePoint(f"task-{task_id}: train", "1")
                tp.begin()
                await asyncio.gather(*self.tasks[task_id].actor_model.async_train(rollout_id))
                tp.end()
                end_time = time.time()
                Timer().add(task_id, "task-train", end_time-start_time)

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
                    start_time = time.time()
                    tp = TracePoint(f"task-{task_id}: actor model offload", "1")
                    tp.begin()
                    await asyncio.gather(*(self.tasks[task_id].actor_model.async_offload()))
                    tp.end()
                    end_time = time.time()
                    Timer().add(task_id, "task-postprocess-train", end_time-start_time)

        wandb.finish()
        Timer().end(task_id, "task-runtime")