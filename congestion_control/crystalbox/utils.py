import numpy as np
from typing import Union, List, Dict, TypedDict, Tuple, Callable
from gymnasium.utils import seeding
from pathlib import Path


class RandomActionGenerator:
    """A wrapper to sample epsilon greedy actions. 
        Used by passing true actions to each sample call, and having them 
        replaced with random actions with epsilon probability
    """

    def __init__(self, possible_actions: List[List], 
                 replace_prob: float,
                 seed: int = 13):
        """Create and initialize the prng for random actions

        Args:
            possible_actions: List of all possible random actions
                defined as a 2D list of [original_action, possible candidates]  
                where dim 0 implicitly acts as the original action
            replace_prob: The epsilon probability to replace true actions with
            seed: The prng seed. Defaults to 13.
        """
        self.possible_actions = np.array(possible_actions)
        self.n_possible_actions = self.possible_actions.shape[1]
        self.np_random, _ = seeding.np_random(seed)
        self.replace_prob = replace_prob

    def sample(self, actions: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
        """
        Sample n_actions from the possible actions, with probability self.replace_prob 
        to replace. Actions not replaced are None

        Args:
            actions: The actions to be already taken by the agent. 
                Used to sample possible alternative actions with. Must be already batched
            mask: A boolean array that specifies which actions to not replace even if sampled.
                Indices corresponding to False values will NOT be replaced
        Returns:
            An array of dim n_actions, that either contain random actions or None otherwise. 
        """
        n_actions = actions.shape[0]

        if mask is not None and mask.shape[0] != n_actions:
            raise ValueError("Mask provided does not match n_actions")
        rand_actions = np.full(shape=(n_actions,), fill_value=None, dtype=object)
        if mask is None:
            mask = np.full(shape=(n_actions,), fill_value=True, dtype=bool)
        to_replace = self.np_random.random((n_actions,)) <= self.replace_prob
        to_replace = to_replace & mask
        rand_indices = self.np_random.choice(
            self.n_possible_actions, size=(n_actions,), replace=True)
        possible_rand_actions = self.possible_actions[actions][np.arange(n_actions) , rand_indices]
        rand_actions[to_replace] = possible_rand_actions[to_replace]
        return rand_actions

class ContinuousRandomActionGenerator(RandomActionGenerator):
    """The continuous counterpart to Random Action Generator. 
        First discretizes the original actions with predefined bins 
        and performs the same replacement with possile actions
    """
    def __init__(self, bins: List, 
            possible_actions: List[List], replace_prob: float,
            seed: int = 13):
        """Create and initialize the prng for random actions

        Args:
            bins: The bins to discretize the original actions with. 
                The indices of these bins act as the original actions 
                in the replacement procedure
            possible_actions: List of all possible random actions
                defined as a 2D list of [original_action, possible candidates]  
                where dim 0 implicitly acts as the original action
            replace_prob: The epsilon probability to replace true actions with
            seed: The prng seed. Defaults to 13.

        Raises:
            ValueError: If len(bins) is not 5
        """
        if len(bins) != 5:
            raise ValueError
        self.action_bins = np.array(bins)

        super().__init__(possible_actions=possible_actions, 
            replace_prob=replace_prob, seed=seed)

    def sample(self, actions: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Sample n_actions from the possible actions, with probability self.replace_prob 
        to replace. Actions not replaced are None. 

        Args:
            actions: The actions to be already taken by the agent. 
                Used to sample possible alternative actions with. Must be already batched
            mask: A boolean array that specifies which actions to not replace even if sampled.
                Indices corresponding to False values will NOT be replaced
        Returns:
            An array of dim n_actions, that contains that potentially contains random actions. 
        """
        orig_shape = actions.shape
        discrete_actions = np.digitize(np.abs(actions), bins=self.action_bins, right=False)
        decrease_actions = actions < 0
        discrete_actions[decrease_actions] -=2 
        discrete_actions = np.clip(discrete_actions, 0, len(self.action_bins)-1)
        discrete_actions = discrete_actions.reshape((-1, ))
        return super().sample(actions=discrete_actions, mask=mask).reshape(orig_shape)
    

def discount_n_step(x: np.ndarray, n_step: int, gamma: float) -> np.ndarray:
    """
    Taken from RLlib: https://github.com/ray-project/ray/blob/66650cdadbbc19735d7be4bc613b9c3de30a44da/rllib/evaluation/postprocessing.py#L21
    Args:
        x: The array of rewards 
        n_step: The number of steps to look ahead and adjust.
        gamma: The discount factor.

    Examples:
        n-step=3
        Trajectory=o0 r0 d0, o1 r1 d1, o2 r2 d2, o3 r3 d3, o4 r4 d4=True o5
        gamma=0.9
        Returned trajectory:
        0: o0 [r0 + 0.9*r1 + 0.9^2*r2 + 0.9^3*r3] d3 o0'=o3
        1: o1 [r1 + 0.9*r2 + 0.9^2*r3 + 0.9^3*r4] d4 o1'=o4
        2: o2 [r2 + 0.9*r3 + 0.9^2*r4] d4 o1'=o5
        3: o3 [r3 + 0.9*r4] d4 o3'=o5
        4:
    """
    returns = np.array(x, copy=True, dtype=np.float64)
    len_ = returns.shape[0]
    # Change rewards in place.
    for i in range(len_):
        for j in range(1, n_step):
            if i + j < len_:
                returns[i] += (
                    (gamma ** j) * returns[i + j]
                )
    return returns

def discount_n_step_2d(rewards: np.ndarray, n_step: int, gamma: float) -> np.ndarray:
    """The 2D counterpart to discounting, where rewards are batched across environments.

    Args:
        rewards: The original rewards, in dim[n_envs, n_steps]
        n_step: The maximum n_steps to discount for. 
            Rewards after n_step ignored during discounting
        gamma: The discount factor

    Raises:
        ValueError: If the rewards are not already batched

    Returns:
        The discounted N-step returns
    """
    if rewards.ndim != 2:
        raise ValueError("Passed in rewards not in shape (batch, n_steps..)")
    gammas = gamma ** np.arange(n_step)
    returns = rewards[:, :n_step] @ gammas
    return returns

def compute_discounted_threshold(reward: float, n_step: int, gamma: float) -> float:
    """
    Calculate the n_step threshold associated with the reward, 
        assuming that you'd get this exact reward for n_steps.
    
    Args:
        reward: The undiscounted reward to calculate threshold for.
        n_step: How many steps we'd get this reward for
        gamma: The discount for the n_step
    
    Returns:
        The threshold for the discount
    """
    shape = n_step + 1
    rewards = np.array([reward]*shape)
    return discount_n_step(rewards, n_step, gamma)[0]


def nan_array_generator(shape) -> np.ndarray:
    """Generate a float64 numpy array of nan in shape

    Args:
        shape: the shape of the desired array

    Returns:
        The nan-filled array
    """
    return np.full(shape=shape, fill_value=np.nan, dtype=np.float64)

def create_minmax_scaler(scale: float, max: float) -> Tuple[Callable, Callable]:
    """Create a min/max scaling function that normalizes and unnormalizes the values

    Args:
        scale: the multiplicative scaler to normalize with
        max: the maximum value to clip to

    Returns:
        the callables[normalize, unnormalize] that take array-like values
    """
    def normalizer(value: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        value = value * scale
        value = value / max
        value = np.clip(value, 0., 1.) # numerical stability
        return value

    def unnormalizer(value: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        value = value * max
        value = value / scale
        return value

    return normalizer, unnormalizer

class EpisodicData(TypedDict):
    observations: List[np.ndarray]
    encoded_observations: List[np.ndarray]
    actions: List[float]
    original_actions: List[float]
    is_action_random: List[bool]
    previous_sending_rates: List[float]
    throughputs: List[float]
    latencies: List[float]
    losses: List[float]
    cluster_label: int


def episodic_dict_generator() -> EpisodicData:
    """Create a dictionary containing all information for an episode in the dataset

    Returns:
        Initialized EpisodicData dictionary with empty lists or default values. 
    """
    return EpisodicData(observations= [],
                       encoded_observations= [],
                       actions= [],
                       original_actions= [],
                       is_action_random= [],
                       previous_sending_rates= [],
                       throughputs= [],
                       latencies= [],
                       losses= [], 
                       cluster_label= -1)


def save_dataset(dataset: Dict[int, List[EpisodicData]], dataset_dir: Path) -> None:
    """Save the created standard dataset to directory. Each trace is saved as
        tace_{trace_idx}.npz where trace_idx is read from the dataset

    Args:
        dataset: the dataset to save, a mapping from trace idx to EpisodicData
        dataset_dir: the path to save the traces to.
    """
    name_glob = "trace_{trace_idx}.npz"
    dataset_dir.mkdir(exist_ok=True, parents=True)
    for trace_idx, eps in dataset.items():
        ep_data = {}
        for i, ep in enumerate(eps):
            for key, value in ep.items():
                ep_data[f"{key}__{i}"] = value
        name = name_glob.format(trace_idx=trace_idx)
        np.savez(dataset_dir / name, **ep_data)

def structured_dataset_generator() -> List[List]:
    """Create a list of datasets where idx 0 represents the original policy
        and idx 1 represents the random action (for event detection)

    Returns:
        The structured dataset list
    """
    return [[], []]

def save_differentiated_dataset(dataset: Dict[int, List[List[EpisodicData]]],
                                 dataset_dir: Path) -> None:
    """Save the structured differentiated dataset (for event detection). 
        Original traces are saved as trace_{trace_idx}.npz while random action
        traces are saved as diff__trace_{trace_idx}.npz 

    Args:
        dataset: the dataset to save, a mapping from trace idx to 
            the structured dataset generator output 
        dataset_dir: the directory to save the traces to.
    """
    name_glob = "trace_{trace_idx}.npz"
    dataset_dir.mkdir(exist_ok=True, parents=True)
    for trace_idx, differentiated_eps in dataset.items():
        ep_data = {}
        for kind_eps_idx, eps in enumerate(differentiated_eps):
            ep_prefix = "diff__" if kind_eps_idx == 0 else ""
            for key, value in eps[0].items():
                ep_data[f"{ep_prefix}{key}__0"] = value
        name = name_glob.format(trace_idx=trace_idx)
        np.savez(dataset_dir / name, **ep_data)