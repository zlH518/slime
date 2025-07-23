import copy

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Union, Any, Dict

from slime.ray.placement_group import create_placement_groups


@dataclass
class SchedulerParams:
    """The Params about Scheduler"""    
    tasks_num: int = 1
    tasks_args_dir: str = "xxx"
    tasks_args: List = None
    tasks_args_template = None

    pgs: dict = None
    train_semaphore: int = 1
    rollout_semaphore:int  = 1
    update_weight_semaphore:int = 1
    save_semaphore: int = 1

    # TODO:should read multi tasks args from different yaml file
    def __post_init__(self, args):
        self.pgs = create_placement_groups(self.tasks_args_template)
        
        tasks_args_template.pgs = self.pgs
        tasks_args_template.tasks_num = self.tasks_num

        self.tasks_args = [copy.deepcopy(self.tasks_args_template) for _ in range(self.tasks_num)]

