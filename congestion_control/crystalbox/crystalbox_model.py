from torch import nn
from torch.utils.data.dataset import Dataset
import random
from pathlib import Path
import torch as th
from tqdm.auto import tqdm
from typing import Callable, Dict, List
import numpy as np
from torch.utils.data import DataLoader
from torch.distributions import Normal
import time

N_SAMPLES = 30

from utils import (discount_n_step_2d, compute_discounted_threshold, 
                              discount_n_step, create_minmax_scaler)
from global_constants import (
    N_STEP, GAMMA, NORMALIZATION_WEIGHTS,
    SHUFFLE_SEED, MAX_VALS, N_ACTIONS)


class RewardDataset(Dataset):
    """Load and preprocess the saved reward dataset for training or testing
    """
    def __init__(self, reward_data_dir: Path, split: str, 
            normalize_rewards: bool = True, differentiated: bool = False):
        """Load and preprocess the dataset into memory

        Args:
            reward_data_dir: the path to the directory of the saved dataset
            split: which split ot load data for, one of [TRAIN, VALIDATION, TEST]
            normalize_rewards: Whether to normalize the reward by scaling. Defaults to True.
            differentiated: Whether or not the reward_data_dir 
                points to the differentiated version of the dataset. Defaults to False.
        """
        self.split = split.upper()
        if normalize_rewards:
            self.create_normalizers()
        self.load_episodic_data(reward_data_dir=reward_data_dir, 
            normalize_rewards=normalize_rewards, differentiated=differentiated)

    def load_episodic_data(self, reward_data_dir: Path, 
            normalize_rewards: bool, differentiated: bool) -> None:
        """Load and process the whole dataset

        Args:
            reward_data_dir: the path to the directory of the saved dataset
            normalize_rewards: Whether to normalize the reward by scaling. Defaults to True.
            differentiated: Whether or not the reward_data_dir 
                points to the differentiated version of the dataset. Defaults to False.
        """
        self.episodic_data = []
        self.episode_lens = []
        files = sorted(reward_data_dir.iterdir())
        random.seed(SHUFFLE_SEED)
        random.shuffle(files)
        # select a subset of traces if necessary
        n = int(len(files) * 0.25)
        if self.split == "TRAIN":
            files = files[n:]
        elif self.split == "VALIDATION":
            files = files[:n]
        
        if not files:
            raise ValueError("Did not find any episodic data in the dataset path specified")
        # for every trace, load and process the states, actions and reward data
        for file in tqdm(files, desc=f"{self.split} Files", leave=False):
            all_data = np.load(file)
            episodes_available = []
            i = 0
            while True:
                keys = list(all_data.keys())
                if "__" not in keys[0]:
                    episodes_available.append("")
                    break
                if np.any([key.endswith(f"__{i}") for key in keys]):
                    episodes_available.append(f"__{i}")
                    i += 1
                else:
                    break
            # check if it's the differentiated dataset
            for ep_ending in episodes_available:
                all_keys = [key for key in all_data.keys() if key.endswith(ep_ending)]
                if np.any([key.startswith("diff__") for key in all_keys]):
                    diff_ep_prefix = "diff__"
                    if differentiated:
                        all_keys = [key for key in all_keys if key.startswith(diff_ep_prefix)]
                    else:
                        all_keys = [key for key in all_keys if not key.startswith(diff_ep_prefix)]
                data = {key.removesuffix(ep_ending):all_data[key] for key in all_keys}
                if differentiated and np.any([key.startswith("diff") for key in all_keys]):
                    data = {key.removeprefix("diff__"): value for key, value in data.items()}

                ep_data = {}
                # calculate and story returns
                for key in NORMALIZATION_WEIGHTS:
                    if len(data[key].shape) == 1:
                        returns = discount_n_step(data[key], n_step=N_STEP, gamma=GAMMA)
                    else:
                        returns = discount_n_step_2d(data[key], n_step=N_STEP, gamma=GAMMA)
                    
                    valid_indices = ~np.isnan(returns)
                    ep_data[key] = returns[valid_indices]
                    
                if normalize_rewards:
                    for key, normalizer in self.normalizers.items():
                        ep_data[key] = normalizer(ep_data[key])
                        ep_data[key] = ep_data[key].reshape((-1, 1))

                ep_data["observations"] = data["observations"][valid_indices]
                ep_data["encoded_observations"] = data["encoded_observations"][valid_indices]
                
                ep_data["previous_sending_rates"] = data["previous_sending_rates"][valid_indices]
                ep_data["is_action_random"] = data["is_action_random"][valid_indices]
                ep_data["original_actions"] = data["original_actions"][valid_indices]
                cluster_labels = np.array([data["cluster_label"]] * data["actions"].shape[0], dtype=np.int64)
                ep_data["cluster_labels"] = cluster_labels[valid_indices]

                ep_data["actions"] =data["actions"][valid_indices]
                ep_len = ep_data["actions"].shape[0]
                self.episodic_data.append(ep_data)
                self.episode_lens.append(ep_len)
        self.episode_lens = np.cumsum(np.asarray(self.episode_lens))
    
    def __len__(self) -> int:
        """Get the length of the loaded data. Used by Dataloader

        Returns:
            the length
        """
        return self.episode_lens[-1]

    def __getitem__(self, idx: int) -> Dict[str, th.Tensor]:
        """Obtain the item at idx of the dataset. Used by Dataloader. 

        Args:
            idx: the id of the sample, in range [0, len(self) - 1]

        Returns:
            the sample, a dict with keys [observations, encoded_observations, 
                previous sending_rates, is_action_random, original_actions, 
                cluster_labels, actions]
        """
        ep_idx = np.digitize(idx, bins=self.episode_lens, right=False)
        last_ep_idx = np.clip(ep_idx -1, 0, self.episode_lens.shape[0])
        step_idx = idx - self.episode_lens[last_ep_idx]
        ep_data = {key: self.episodic_data[ep_idx][key][step_idx] for key in self.episodic_data[ep_idx].keys()}
        data = {}
        for key, val in ep_data.items():
            if key in ["cluster_labels"]:
                dtype = th.long
            elif key in ["is_action_random"]:
                dtype = th.bool
            else:
                dtype= th.float32
            data[key] = th.as_tensor(val, dtype=dtype)
        return data

    def calculate_weights(self, 
            key: str, value: float, target_weight: float = 0.5) -> np.ndarray:
        """Calculate the sampling weights of the states such that when they
            are used by a random weighted sampler, the sampling probability of key
            with value is equal to target_weight

        Args:
            key: the key in episodic_data to use
            value: the value that key should have
            target_weight: the sampling probability that value should have. Defaults to 0.5.

        Returns:
            the sampling weights as a numpy array, of shape [len(self), ]
        """
        n_steps = self.episode_lens[-1]
        weights = np.zeros(shape=(n_steps,), dtype=np.float64)
        labels = np.full_like(weights, dtype=bool, fill_value=False)
        for idx in range(weights.shape[0]):
            ep_idx = np.digitize(idx, bins=self.episode_lens, right=False)
            last_ep_idx = np.clip(ep_idx -1, 0, self.episode_lens.shape[0])
            step_idx = idx - self.episode_lens[last_ep_idx]
            is_label = self.episodic_data[ep_idx][key][step_idx] == value
            labels[idx] = is_label
        n_instances = np.sum(labels)
        print(f"Adjusting weight from {n_instances/n_steps:.5f} to {target_weight}.")
        pos_weight = target_weight / n_instances
        neg_weight = (1. - target_weight) / (n_steps - n_instances)
        weights[labels == True] = pos_weight
        weights[labels == False] = neg_weight
        weights /= weights.sum() # ensure numerical stability
        return weights

    def create_normalizers(self) -> None:
        """initialize all the normalizers/unnormalizers for the dataset
        """
        self.normalizers = {}
        self.unnormalizers = {}
        for key, scale in NORMALIZATION_WEIGHTS.items():
            max = MAX_VALS[key]
            max = compute_discounted_threshold(max, N_STEP, GAMMA)
            normalizer, unnormalizer = create_minmax_scaler(scale, max)
            self.normalizers[key] = normalizer
            self.unnormalizers[key] = unnormalizer



class PredictorTail(nn.Module):
    """Take the shared features and predict a single reward component 
        return using a 1-layer MLP
    """
    def __init__(self, input_dim: int, shared_dim: int = 32):
        """Initialize the neural network

        Args:
            input_dim: the dim of the original tensor
            shared_dim: the size of the single layer. Defaults to 32.
        """
        super(PredictorTail, self).__init__()
        self.shared_net = nn.Sequential(
            nn.Linear(input_dim, shared_dim), 
            nn.ELU(),
            nn.Linear(shared_dim, 75),
            nn.ELU())
        self.mean_predictor = nn.Sequential(
            nn.Linear(75, 1)
        )
        self.std_predictor = nn.Sequential(
            nn.Linear(75, 1),
            nn.Softplus()
        )

    def forward(self, data: th.Tensor) -> Normal:
        """Forward pass through the nn

        Args:
            data: the shared features

        Returns:
            the predicted return, output as a normal distribution
        """
        features = self.shared_net(data)
        mean, std = self.mean_predictor(features), self.std_predictor(features)
        dist = Normal(mean, std)
        return dist


class QConverter:
    """A wrapper to convert the predicted components pack to a numeral return value
    """
    def __init__(self) -> None:
        self.unnormalizers = {}
        for key, scale in NORMALIZATION_WEIGHTS.items():
            max = MAX_VALS[key]
            max = compute_discounted_threshold(max, N_STEP, GAMMA)
            normalizer, unnormalizer = create_minmax_scaler(scale, max)
            self.unnormalizers[key] = unnormalizer

    def __call__(self, output: Dict[str, th.Tensor]) -> th.Tensor:
        """Convert the output of the model to a numerical value
        Args:
            output: The output of model, a  dict with reward components and the associated distributions

        Returns:
            the effective state-action value, a tensor of dim [batch, 1]
        """
        key = next(iter(self.unnormalizers))
        returns = th.zeros_like(output[key].mean, dtype=th.float32)
        for key in self.unnormalizers:
            returns += self.unnormalizers[key](output[key].mean) 
        return returns


class RewardNetwork(nn.Module):
    """The core predictor network
    """
    def __init__(self, normalize_rewards: bool = True, 
            action_dim: int = N_ACTIONS, input_dim: int = 128, 
            shared_dim = 75):
        """Initialize the predictor network. It transforms the embedding to 
            a shared feature space, and then uses a split MLP that predict 
            the mean/std of each of the reward components.

        Args:
            normalize_rewards: whether to scale and clip the returns. Defaults to True.
            action_dim: dim 0 of the actions. Defaults to N_ACTIONS.
            input_dim: the embedding size of the controller. Defaults to 256.
            shared_dim: the shared feature space for the predictor. Defaults to 75.
        """
        super(RewardNetwork, self).__init__()

        self.shared_net = nn.Sequential(
            nn.Linear(input_dim + action_dim, shared_dim),
            nn.ELU()
        )
        self.throughput_predictor = PredictorTail(input_dim=shared_dim+action_dim, shared_dim=shared_dim)
        self.latency_predictor = PredictorTail(input_dim=shared_dim+action_dim, shared_dim=shared_dim)
        self.loss_predictor = PredictorTail(input_dim=shared_dim+action_dim, shared_dim=shared_dim)
        self.unnormalizers = None
        if normalize_rewards:
            self.unnormalizers = {}
            for key, scale in NORMALIZATION_WEIGHTS.items():
                max = MAX_VALS[key]
                max = compute_discounted_threshold(max, N_STEP, GAMMA)
                normalizer, unnormalizer = create_minmax_scaler(scale, max)
                self.unnormalizers[key] = unnormalizer
        self.q_converter = QConverter()

    def forward(self, data: Dict[str, th.Tensor]) -> dict:
        """A forward pass through the neural network. 
        The input is expected to the dictionary from the dataset.

        Args:
            data: the output of RewardDataset[idx]

        Returns:
            The predicted distribution of future returns,
                and the effective return in key 'returns'
        """
        encoded_observations = data["encoded_observations"]
        features = self.shared_net(th.cat((encoded_observations, data["actions"]), dim=1))
        features = th.cat((features, data["actions"]), dim=1)
        output = {}
        for key, tail in [["throughputs", self.throughput_predictor],
                        ["latencies", self.latency_predictor], 
                        ["losses", self.loss_predictor]]:
            dist = tail(features)
            if not self.training:
                dist = Normal(th.clamp(dist.mean, 0, 1), dist.stddev)
            output[key] = dist
        output["returns"] = self.q_converter(output)
        return output


def init_weights(model: nn.Module) -> None:
    """Wrapper for initializing the weights using kaiming uniform

    Args:
        model: the model to initialize
    """
    if np.any([isinstance(model, kind_layer) for kind_layer in [nn.Linear, nn.Conv1d]]):
        if model.weight.requires_grad:
            nn.init.kaiming_uniform_(model.weight)
            model.bias.data.fill_(0.0001)


def dfs_freeze(model: nn.Module) -> None:
    """Wrapper to freeze the gradients for a specific model and its children

    Args:
        model: the model to freeze
    """
    for name, child in model.named_children():
        for param in child.parameters():
            param.requires_grad = False
        dfs_freeze(child)
    

def calculate_runtimes(dataset: RewardDataset, algorithm: Callable, 
                       device: str, 
                       n_samples = 500) -> List[float]:
    """Calculate the per-sample latency of the predictor, measuring it across n_samples

    Args:
        dataset: the test dataset to evaluate the runtimes with
        algorithm: the model/algorithm to benchmark
        device: the device to use, if using Torch and GPU
        n_samples: the number of samples to calculate latency for. Defaults to 500.

    Returns:
        the runtime latency of each sample, a list of shape [n_samples, ]
    """
    run_times = []

    test_dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    with th.no_grad():
        for data in test_dataloader:
            if len(run_times) >= n_samples:
                break
            for key in data:
                data[key]= data[key].to(device)
            start_time = time.time()
            algorithm(data)
            end_time = time.time()
            run_times.append(end_time - start_time) # in seconds
    return run_times