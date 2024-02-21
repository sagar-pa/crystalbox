import numpy as np
from torch.nn.functional import mse_loss
from torch.utils.data import DataLoader
import torch as th
from tqdm.auto import tqdm
from typing import Callable, Dict, Tuple, List
import pickle as pk
from gymnasium.utils.seeding import np_random
from stable_baselines3.common.vec_env import VecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3 import A2C
from argparse import ArgumentParser

from cc_rl.trace_loader import load_trace_features
from cc_rl.network_sim import SimulatedNetworkEnv
from global_constants import (
    CRYSTALBOX_DATA_SAVE_PATH,  
    NORMALIZE_REWARDS, SEED, N_ENVS, MODEL_PATH, BATCH_SIZE,
    TEST_DATASET_DIR, DIFF_DATASET_DIR, INFO_LOOKUP, WEIGHTS, NORMALIZATION_WEIGHTS,
    WEIGHTS, MAX_VALS, N_STEP, GAMMA, DEVICE)
from utils import (
    discount_n_step_2d, compute_discounted_threshold, create_minmax_scaler)
from crystalbox_model import (
    RewardDataset, calculate_runtimes)



def stack_3d(history: np.ndarray, new_data: np.ndarray)-> np.ndarray:
    """
    Stack the arrays with the new data.
    Args:
        history: Array of shape (batch, history_len, data_len)
        new_data: Array of shape (batch, data_len)
    Returns:
        New array with new_data added, of same shape as history
    """
    if not history.ndim == 3 or not new_data.ndim == 2:
        raise ValueError("History Must be 3D array already batched, and new_data must also be batched")
    new_data = new_data[:, np.newaxis, :]
    
    new_obs = np.concatenate((history, new_data), axis=1)
    return new_obs[:, 1:, :]

def test(dataloader: DataLoader, model: Callable, device: str, split: str
         ) -> Tuple[Dict[str, List], Dict[str, List]]:
    """Evaluate the sampling baselines with a given dataloader

    Args:
        dataloader: the test dataloader to evaluate with
        model: the sampling algorithm to use
        device: which device to use to calculate loss
        split: which split the dataloader belongs to. Only used for printing

    Returns:
        the (mse, predictions) of the model
    """
    tqdm.write(f"------ Testing with {split} Split ------")
    mse_losses= {
        "throughputs": [],
        "latencies": [],
        "losses": []
    }
    preds = {
        "throughputs": [],
        "latencies": [],
        "losses": []
    }

    with th.no_grad():
        for data in tqdm(dataloader, desc="Testing Batches", leave=False):
            for key in data:
                data[key]= data[key].to(device)
            pred = model(data)
            for key in mse_losses:
                samples = th.as_tensor(pred[key], device=device).reshape(-1, 1)
                preds[key].extend(samples.detach().cpu().tolist())
                mse = mse_loss(samples, data[key], reduction="none").flatten()
                mse_losses[key].extend(mse.detach().cpu().tolist())


    for key in mse_losses:
        mse = np.round(np.mean(mse_losses[key]), 6)
        tqdm.write(f"{key:25} MSE: {mse}", end="\n")

    return mse_losses, preds


def simulate(
        data: Dict[str, th.Tensor],
        idx: int,
        model: BaseAlgorithm,
        env: VecEnv,
        rand: np.random,
        cluster_labels: np.ndarray,
        use_weighted: bool = False,
        n_step = 5,
        gamma = 0.9
    ) -> Tuple[int, Dict[str, float]]:
    """Calculate the predicted returns based on a sampling approach.
        Performs the simulation for a specific idx in the whole batch data.

    Args:
        data: the whole batch of obs/action data
        idx: the specific idx in the data batch to process
        model: the controller to query for actions
        env: the environment to perform sampling with
        rand: the seeded prng module (for reproducibility)
        cluster_labels: the labels for the trace idx. 
            Used to perform distribution-aware sampling if flag is set 
        use_weighted: whether to use distribution-aware sampling. Defaults to False.
        n_step: the horizon of the simulation. Defaults to 5.
        gamma: the discount factor. Defaults to 0.9.


    Returns:
        the given idx, and the predictions of each reward component
    """
    
    traces_to_sample = np.arange(cluster_labels.shape[0])
    cluster_label = data["cluster_labels"][idx].item()
    num_traces = env.num_envs

    if use_weighted:
        candidate_traces = cluster_labels == cluster_label
        traces_to_sample = traces_to_sample[candidate_traces]
    sampled_trace_indices = rand.choice(traces_to_sample.shape[0],
        size=(num_traces,), replace=True)
    sampled_traces = traces_to_sample[sampled_trace_indices]

    observation = data["observations"][idx].detach().cpu().numpy()
    action = np.array(data["actions"][idx].item())
    previous_sending_rate = data["previous_sending_rates"][idx].item()

    observations = np.repeat(observation[np.newaxis, ...], num_traces, axis=0) 
    actions = np.repeat(action[np.newaxis, ...], num_traces, axis=0)
    
    rewards = dict(
        throughputs = np.zeros((num_traces, n_step), dtype=np.float64),
        latencies = np.zeros((num_traces, n_step), dtype=np.float64),
        losses = np.zeros((num_traces, n_step), dtype=np.float64)
    )

    for rank, trace_idx in enumerate(sampled_traces):
        env.env_method("set_sampler_attr", "idx", trace_idx-1, indices=rank)
    env.reset()
    env.env_method("set_sending_rate", previous_sending_rate)
    
    for step_idx in range(n_step):
        next_observations, _, __, infos = env.step(actions)
        for env_idx in range(num_traces):
            for key, weight in WEIGHTS.items():
                rewards[key][env_idx, step_idx] = infos[env_idx][INFO_LOOKUP[key]] * weight
            if infos[env_idx]["trace_idx"] != sampled_traces[env_idx]:
                raise ValueError("Sampled trace not selected by env!")
        observation = next_observations[:, -1, :]
        observations = stack_3d(observations, observation)
        actions, *_ = model.predict(observations)
    
    output = {}
    for key in rewards:
        returns = discount_n_step_2d(rewards[key], n_step=n_step, gamma=gamma)
        output[key] = np.mean(returns)
        output[f"{key}_std"] = np.std(returns)
    
    return (idx, output)


class MCSampler:
    """A wrapper for the simulate function that handles the batching logic
    """
    def __init__(self,
            model: BaseAlgorithm,
            env: VecEnv,
            cluster_labels: np.ndarray,
            seed: int = 12,
            use_weighted: bool = False,
            n_step: int = 5,
            gamma = 0.9,
            normalize_rewards: bool = NORMALIZE_REWARDS
        ) -> None:
        """Initialize the wrapper

        Args:
            model: the controller to query for actions
            env: the environment to perform sampling with
            cluster_labels: the labels for the trace idx. 
                Used to perform distribution-aware sampling if flag is set
            seed: the prng seed for sampling. Defaults to 12.
            use_weighted: whether to use distribution-aware sampling. Defaults to False.
            n_step: the horizon of the simulation. Defaults to 5.
            gamma: the discount factor. Defaults to 0.9.
            normalize_rewards: whether to scale and clip the returns. 
                Defaults to NORMALIZE_REWARD.
        """

        self.model = model
        self.env = env
        self.cluster_labels = cluster_labels
        self.rand, *_ = np_random(seed=seed)
        self.use_weighted = use_weighted
        self.n_step = n_step
        self.gamma = gamma
        self.normalize_rewards = normalize_rewards
        if normalize_rewards:
            self.create_normalizers()

    def __call__(self, data: dict[str, th.Tensor]) -> dict[str, np.ndarray]:
        """A effective forward pass through the sampling based approach

        Args:
            data: the batch of data to process, as given by the RewardDataset

        Returns:
            the predicted returns for that batch, a dict with
                reward component as keys and arrays of dim [batch_size, ] as values
        """
        batch_size = data["observations"].shape[0]
        output = {}
        for key in WEIGHTS:
            output[key] = np.zeros((batch_size, ), dtype=np.float64)
            output[f"{key}_std"] = np.zeros((batch_size, ), dtype=np.float64)

        for i in range(batch_size):
            _, sampling_output = simulate(
                data = data,
                idx = i,
                model = self.model,
                env = self.env,
                rand = self.rand,
                cluster_labels = self.cluster_labels,
                use_weighted = self.use_weighted,
                n_step = self.n_step,
                gamma = self.gamma
            )
            for key, val in sampling_output.items():
                output[key][i] = val

        if self.normalize_rewards:
            for key, normalizer in self.normalizers.items():
                output[key] = normalizer(output[key])

        return output


    def create_normalizers(self) -> None:
        """Initialize the normalization to be applied to the output
        """
        self.normalizers = {}
        self.unnormalizers = {}
        for key, scale in NORMALIZATION_WEIGHTS.items():
            max = MAX_VALS[key]
            max = compute_discounted_threshold(max, N_STEP, GAMMA)
            normalizer, unnormalizer = create_minmax_scaler(scale, max)
            self.normalizers[key] = normalizer
            self.unnormalizers[key] = unnormalizer
            
def evaluate_sampling_baseline() -> None:
    train_features = load_trace_features(split="train")
    train_labels = train_features.cluster_labels
    
    train_env_kwargs = dict(sampler_kwargs=dict(split="train", sampling_func_cls="iterative"), 
        is_action_continuous=True)
    train_env = make_vec_env(SimulatedNetworkEnv, n_envs=N_ENVS, seed=SEED, 
        env_kwargs=train_env_kwargs, vec_env_cls=SubprocVecEnv,
        vec_env_kwargs=dict(start_method="forkserver"))        
    model = A2C.load(MODEL_PATH)
    
    test_dataset = RewardDataset(TEST_DATASET_DIR, "test")
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    filtered_test_dataset = RewardDataset(DIFF_DATASET_DIR, "test")
    filtered_test_dataloader = DataLoader(filtered_test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    differentiated_test_dataset = RewardDataset(DIFF_DATASET_DIR, "test", differentiated=True)
    differentiated_test_dataloader = DataLoader(differentiated_test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    for is_weighted, name in [[False, ""], [True, "weighted"]]:
        print(f"{'-' * 15} Testing sampling with weighted={is_weighted} {'-' * 15}")
        sampler = MCSampler(model=model, 
                            env=train_env, 
                            seed=SEED, 
                            cluster_labels=train_labels, 
                            use_weighted=is_weighted, 
                            n_step=N_STEP, 
                            gamma=GAMMA)
        for dataset_save_path, dataloader in [
                [f"sampling_test_mses_{name}.pk", test_dataloader],
                [f"sampling_filtered_mses_{name}.pk", filtered_test_dataloader],
                [f"sampling_perturbed_mses_{name}.pk", differentiated_test_dataloader]
            ]:
            test_mse, preds = test(dataloader, sampler, DEVICE, "Test")
            with open(CRYSTALBOX_DATA_SAVE_PATH / dataset_save_path, "wb") as f:
                pk.dump(dict(mses=test_mse, preds=preds), f)
    
    #extra evaluation stats
    run_times = calculate_runtimes(dataset= test_dataset, algorithm=sampler, device=DEVICE)
    save_path = CRYSTALBOX_DATA_SAVE_PATH / "cc_sampling_runtimes.pk"
    with open(save_path, "wb") as f:
        pk.dump(dict(run_times=run_times), f)
        
    train_env.close()
    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--test", action="store_true", default=False,
                        help="Whether to test the sampling baseline on the dataset")
    args = parser.parse_args()
    if args.test:
        evaluate_sampling_baseline()