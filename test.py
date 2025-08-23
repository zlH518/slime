import time
import socket
import ray
import abc

from typing import Any, Callable, Dict, List, Tuple
from ray.util.placement_group import PlacementGroup
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from slime.ray.placement_group import _create_placement_group

# --- Configuration ---
NUM_TASKS = 3
TOTAL_GPU_MEM = 16 * 1024  # 16 GB in MB
MAX_CONCURRENT_PER_EVENT = {
    'update': 1,
    'inference': 2,
    'train': 1,
}
NUM_CYCLES_PER_TASK = 3


class Profile:
    def __init__(self) -> None:
        pass

    def recode(self):
        pass


class Scheduler:
    _request_queue = []
    _task = {}

    def __init__(self) -> None:
        pass

    def schedule(self):
        pass
    

class TrainActor:
    def __init__(self, world_size, rank, master_addr, master_port, global_rank,) -> None:

        if master_addr is None or master_port is None:
            master_addr, master_port = ray.util.get_node_ip_address(), 6379
        
        self.world_size = world_size
        self.rank = rank
        self.global_rank = global_rank
        self.master_addr = master_addr
        self.master_port = master_port

    def get_master_addr_and_port(self):
        return self.master_addr, self.master_port

    def train(self):
        pass

    def update(self):
        pass


class RolloutActor:
    def __init__(self) -> None:
        pass

    def rollout(self):
        pass

class TrainActorGroup:
    def __init__(
        self,
        pgs,
        num_nodes,
        num_gpus_per_node,
        num_gpus_per_actors,
        num_cpus_per_actors,
        resources    
    ):
        self.pgs = pgs
        self.world_size = num_nodes * num_gpus_per_node
        self.num_gpus_per_actor = num_gpus_per_actors
        self.num_cpus_per_actor = num_cpus_per_actors
        self.resources = resources

        self.actors_handlers = []
        self._allocate_gpus_for_actors(pgs)

    def _allocate_gpus_for_actors(self, pgs):
        pg, global_offset, reordered_bundle_indices = pgs

        TrainRayActor = ray.remote(
            num_gpus=1,
        )(TrainActor)

        master_addr, master_port = None, None
        for rank in range(self.world_size):
            actor = TrainRayActor.options(
                num_cpus=self.num_cpus_per_actor,
                num_gpus=self.num_gpus_per_actor,
                resources=self.resources,
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_bundle_index=reordered_bundle_indices[rank],
                ),
            ).remote(self.world_size, rank, master_addr, master_port, rank+global_offset)
            if rank == 0:
                master_addr, master_port = ray.get(actor.get_master_addr_and_port.remote())
            self.actors_handlers.append(actor)


class Task:
    _task_id = 0
    def __init__(self, args, pgs) -> None:
        self.task_id = Task._task_id
        Task._task_id += 1

        self.args = args
        self.pgs = pgs

        self.train_actor_group = TrainActorGroup(
            pgs,
            self.args.actor.num_nodes,
            self.args.actor.num_gpus_per_node,
            self.args.actor.num_gpus_per_actors,
            self.args.actor.num_cpus_per_actors,
            self.args.actor.resources  
        )

        self.rollout_actor = RolloutActor()

    


    def run(self):
        for i in range(10):
            pass
            


if __name__ == "__main__":
    # asyncio.run(main())
    pg, actor_pg_reordered_bundle_indices = _create_placement_group(8)

    rollout_pg_reordered_bundle_indices = actor_pg_reordered_bundle_indices[4:]

    pgs = {
        "actor": (pg, 0, actor_pg_reordered_bundle_indices),
        "rollout": (pg, 4, rollout_pg_reordered_bundle_indices),
    }

