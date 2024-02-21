from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import get_linear_fn
from typing import Union, List
import ray

class AnnealingCallBack(BaseCallback):
    def __init__(self, variable: str, start: float, end: float, 
            total_train_steps: Union[int, float], n_train_envs: int, 
            end_fraction: float = 0.5, update_freq: int = 20):
        super(AnnealingCallBack, self).__init__()
        self.variable = variable
        self.function = get_linear_fn(start, end, end_fraction)
        self.update_freq = update_freq
        self.total_calls = max(total_train_steps // n_train_envs, 1)

    def _on_step(self) -> bool:
        if self.n_calls % self.update_freq == 0:
            setattr(self.model, self.variable, 
                self.function(1 - self.n_calls / self.total_calls))
        return True



class SharedProgressCallBack(BaseCallback):
    def __init__(self, actor_name: str, progress_increment_unit: int = 5, 
            increment_on_init: bool = True):
        super(SharedProgressCallBack, self).__init__()
        self.actor_name = actor_name
        self.progress_increment_unit = progress_increment_unit
        if increment_on_init:
            self.increment_shared_train_progress()

    def increment_shared_train_progress(self) -> None:
        actor = ray.get_actor(name=self.actor_name, namespace="test")
        curr_progress = ray.get(actor.get_train_progress.remote())
        new_progress = curr_progress + self.progress_increment_unit
        actor.set_train_progress.remote(new_progress)

    def _on_step(self) -> bool:
        self.increment_shared_train_progress()
        return True
