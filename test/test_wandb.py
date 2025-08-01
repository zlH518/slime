import os
import ray
import wandb

os.environ["WANDB_MODE"] = "offline"
os.environ["WANDB_API_KEY"] = "dummykey1234567890"
os.environ["WANDB_DIR"] = "/volume/pt-train/users/mingjie/hzl_code/code/slime/test"

GROUP_NAME = "731"  # 你可以用时间戳、实验名等

@ray.remote
class Worker:
    def __init__(self, rank):
        self.rank = rank
        class_name = self.__class__.__name__
        self.run = wandb.init(
            project="wandb-dist-demo",
            group=GROUP_NAME,
            name=f"{class_name}{rank}",
            config={"rank": rank},
        )

    def log(self):
        for step in range(5):
            wandb.log({"rank": self.rank, "step": step, "value": self.rank * 10 + step})
            print(f"[rank {self.rank}] step {step} logged")
        wandb.finish()
        return self.run.id

if __name__ == "__main__":
    ray.init(
        runtime_env={
            "env_vars": {
                "WANDB_MODE": "offline",
                "WANDB_API_KEY": "dummykey1234567890",
                "WANDB_DIR": "/volume/pt-train/users/mingjie/hzl_code/code/slime/test"
            }
        }
    )
    workers = [Worker.remote(i) for i in range(4)]
    run_ids = ray.get([w.log.remote() for w in workers])
    print("All run ids:", run_ids)