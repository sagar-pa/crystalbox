import numpy as np
import ray
from ray.util.queue import Queue, Empty
from typing import List, Tuple

def discount_n_step(rewards: np.ndarray, n_step: int, gamma: float) -> float:
    """
    Compute the truncated discounted return of the first state in the reward array.
    Args:
        rewards: the undiscounted rewards to process
        n_step: the maximum horizon to enforce. Rewards after n_step are assumed to be zero.
        gamma: the discount factor
    Returns:
        The discounted return
    """
    if rewards.ndim != 1:
        raise ValueError("Passed in rewards not in shape (n_step..)")
    gammas = gamma ** np.arange(n_step)
    returns = rewards[np.newaxis, :n_step] @ gammas
    return returns[0]

def format_float(num: float, precision: int = 4) -> str:
    """
    Create a string with num with the given precision (by rounding)
    Args:
        num: the number to format
        precision: the precision of the final string
    Returns:
        the string
    """
    return str(round(num, precision))


def terminate_shared_data(actor_ids: List[Tuple[str, str]]) -> None:
    """
    Forcefully terminate actors by their name (useful when ray instances do not die)
    Args:
        actor_ids: a list of name, namespace tuples for actors to terminate
    """
    for name, namespace in actor_ids:
        actor = ray.get_actor(name, namespace)
        ray.kill(actor)


@ray.remote(num_cpus=0.01)
class SharedTestingData:
    def __init__(self, n_traces: int):
        self.train_progress = 0
        self.n_traces = n_traces
        self.shared_testing_indices = Queue(actor_options=dict(num_cpus=0.01))
        self.shared_testing_indices.put_nowait_batch(list(range(n_traces)))
    
    def get_index_to_test(self) -> int:
        try:
            idx = self.shared_testing_indices.get_nowait()
        except Empty:
            idx = np.random.randint(low=0, high=self.n_traces)
        return idx

    def get_train_progress(self) -> int:
        return self.train_progress

    def set_train_progress(self, train_progress: int):
        self.train_progress = int(train_progress)
        while not self.shared_testing_indices.empty():
            try:
                self.shared_testing_indices.get_nowait_batch(len(self.shared_testing_indices))
            except Empty:
                break
        self.shared_testing_indices.put_nowait_batch(list(range(self.n_traces)))