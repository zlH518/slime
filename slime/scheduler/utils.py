import copy

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Union, Any, Dict

from slime.ray.placement_group import create_placement_groups


@dataclass
class SchedulerParams:
    """The Params about Scheduler"""    
    tasks_num: int = 0
    tasks_args_dir: str = "xxx"
    tasks_args: list = None
    tasks_args_template: Any = None

    pgs: dict = None

    # TODO:should read multi tasks args from different yaml file
    def __post_init__(self):
        self.pgs = create_placement_groups(self.tasks_args_template)
        self.tasks_num = self.tasks_args_template.tasks_num
        
        self.tasks_args_template.pgs = self.pgs

        # TODO 读取task_args_dir下的list，然后分别整理成参数列表返回
        self.tasks_args = [copy.deepcopy(self.tasks_args_template) for _ in range(self.tasks_num)]

