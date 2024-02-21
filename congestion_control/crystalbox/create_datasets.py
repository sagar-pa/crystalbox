from collections import defaultdict
import numpy as np
import torch as th
from stable_baselines3 import A2C
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import VecEnv, SubprocVecEnv
from pathlib import Path
from stable_baselines3.common.env_util import make_vec_env
from tqdm.auto import tqdm
from typing import List, Dict

from utils import (ContinuousRandomActionGenerator, 
                              nan_array_generator, structured_dataset_generator,
                              save_differentiated_dataset,
                              EpisodicData, episodic_dict_generator, save_dataset)
from global_constants import (
    WEIGHTS, INFO_LOOKUP, N_ENVS, N_EPS, SEED, CONTINUOUS_ACTION_BINS, 
    CONTINUOUS_POSSIBLE_RAND_ACTIONS, MODEL_PATH, 
    TRAIN_DATASET_DIR, TEST_DATASET_DIR, N_TEST_EPS, 
    DIFF_DATASET_DIR, N_DIFF_EPS, AUXILIARY_REWARD_WEIGHTS, 
    AUXILIARY_DATASET_DIR, AUXILIARY_MODEL_PATH, DEVICE)
from cc_rl.trace_loader import load_trace_features
from cc_rl.network_sim import SimulatedNetworkEnv


def generate_dataset(env: VecEnv,
        model: BaseAlgorithm,
        n_eps: int,
        cluster_labels: np.ndarray,
        action_generator: ContinuousRandomActionGenerator,
        follow_n_steps_after: int = 20,
        decorelating_steps: int = 15,
        suppress_progress: bool = False) -> Dict[int, List[EpisodicData]]:
    """Create a dataset of episodic data using the model and env.
        The dataset contains a [state, action, reward, ... reward + follow_n_steps_after]
        Ignores the first decorelating_steps in the episode.

    Args:
        env: the environment to interact with
        model: the agent policy to rollout
        n_eps: the total number of episodes to have in the dataset
        cluster_labels: the cluster id of the traces where dim 0 is the 
        action_generator: the random action genrator
        follow_n_steps_after: how many steps to follow the policy for after recording action. Defaults to 30.
        decorelating_steps: how many first steps to ingore. Defaults to 5.
        suppress_progress: whether or not to disable tqdm pbar. Defaults to False.

    Returns:
        The gathered dataset
    """
    
    n_envs = env.num_envs
    recorded = np.full(shape=(n_envs, ), fill_value=False, dtype=bool)

    n_traces = cluster_labels.shape[0]
    for rank in range(n_envs):
        env.env_method("set_sampler_attr", "idx", 
                int((rank*n_traces)//n_envs), indices=rank)
    observations = env.reset()
    states = None
    episode_starts = np.ones((env.num_envs,), dtype=bool)
    original_actions = None
    episodic_data = [episodic_dict_generator() for _ in range(n_envs)]
    dataset = defaultdict(list)
    eps_done = 0
    mask = np.array([True] * n_envs)

    cooldowns = np.array([decorelating_steps] * n_envs)
    reward_indices = np.zeros(shape=(n_envs, ), dtype=np.int64)

    with tqdm(total=n_eps, disable=suppress_progress) as pbar:
        while eps_done < n_eps:
            with th.no_grad():
                actions, states = model.predict(
                    observations, state=states, episode_start=episode_starts, deterministic=True)
                obs_tensor = th.as_tensor(observations, dtype=th.float32, device=DEVICE)
                encoded_observations = model.policy.extract_features(obs_tensor).cpu().detach().numpy()

            rand_action_indices = (cooldowns == 1)  & ~recorded
            rand_act_mask = mask & rand_action_indices
            rand_actions = action_generator.sample(actions=actions, mask=rand_act_mask)
            if original_actions is None:
                original_actions = np.zeros_like(actions)
            original_actions[rand_actions != None] = actions[rand_actions != None]
            actions[rand_actions != None] = rand_actions[rand_actions != None]
            next_observations, rewards, dones, infos = env.step(actions)

            for i in range(n_envs):
                done = dones[i]
                info = infos[i]
                episode_starts[i] = done
                trace_idx = info["trace_idx"]
                episodic_data[i]["cluster_label"] = cluster_labels[trace_idx]
                if rand_action_indices[i]: # if this action is supposed to be random
                    recorded[i] = True
                    reward_indices[i] = 0
                    episodic_data[i]["observations"].append(observations[i])
                    episodic_data[i]["encoded_observations"].append(encoded_observations[i])
                    episodic_data[i]["actions"].append(actions[i])
                    episodic_data[i]["previous_sending_rates"].append(info["previous_sending_rate"])
                    episodic_data[i]["original_actions"].append(original_actions[i])
                    episodic_data[i]["is_action_random"].append(rand_actions[i] != None)
                    for key in WEIGHTS:
                        episodic_data[i][key].append(nan_array_generator((follow_n_steps_after, )))
                if len(episodic_data[i]["actions"]) > 0: # if episode is not empty
                    idx = reward_indices[i]
                    if idx < follow_n_steps_after:
                        for key, weight in WEIGHTS.items():
                            episodic_data[i][key][-1][idx] = info[INFO_LOOKUP[key]] * weight
                        reward_indices[i] += 1
                if done: # episode is complete, save data
                    eps_done += 1
                    recorded[i] = False
                    reward_indices[i] = 0
                    cooldowns[i] = decorelating_steps + 1
                    for key in episodic_data[i]:
                        if key in ["actions", "cluster_label"]:
                            dtype = np.int32
                        elif key == "is_action_random":
                            dtype = bool
                        else:
                            dtype = np.float32
                        episodic_data[i][key] = np.asarray(episodic_data[i][key], dtype=dtype)
                    dataset[trace_idx].append(episodic_data[i])
                    episodic_data[i] = episodic_dict_generator()
                    pbar.update(1)
            observations = next_observations
            cooldowns[rand_action_indices] = follow_n_steps_after
            cooldowns[~rand_action_indices] -= 1
            cooldowns = np.clip(cooldowns, 1, follow_n_steps_after)

    return dataset
        
        
ACTION_THRESHOLD = 2
PAST_HORIZON = 4


def generate_differentiated_dataset(
        original_env: VecEnv,
        duplicate_env: VecEnv, 
        model: BaseAlgorithm, 
        n_eps: int,
        cluster_labels: np.ndarray,
        action_generator: ContinuousRandomActionGenerator,
        follow_n_steps_after: int = 20,
        decorelating_steps: int = 15,
        seed: int = SEED,
        suppress_progress: bool = False):
    """Create a dataset of episodic data using the model and env.
        The dataset contains a [state, action, reward, ... reward + follow_n_steps_after]
        Ignores the first decorelating_steps in the episode.
        In this version, the structured dataset contains two version of the episode: 
            one where we follow the policy and another where we have an exploratory first action

    Args:
        original_env: the environment to interact with
        duplicate_env: the duplicate version of the environment to interact with
        model: the agent policy to rollout
        n_eps: the total number of episodes to have in the dataset
        cluster_labels: the cluster id of the traces where dim 0 is the 
        action_generator: the random action genrator
        follow_n_steps_after: how many steps to follow the policy for after recording action. Defaults to 30.
        decorelating_steps: how many first steps to ingore. Defaults to 5.
        suppress_progress: whether or not to disable tqdm pbar. Defaults to False.

    Returns:
        The gathered dataset
    """

    n_envs = original_env.num_envs
    n_traces = cluster_labels.shape[0]
    for rank in range(n_envs):
        original_env.env_method("set_sampler_attr", "idx", 
                int((rank*n_traces)//n_envs), indices=rank)
        duplicate_env.env_method("set_sampler_attr", "idx", 
                int((rank*n_traces)//n_envs), indices=rank)
    original_env.seed(seed)
    duplicate_env.seed(seed)

    all_observations = [env.reset() for env in [duplicate_env, original_env]]
    all_episode_starts = [np.ones((n_envs,), dtype=bool) for _ in range(2)]
    deviated = np.full(shape=(n_envs, ), fill_value=False, dtype=bool)
    all_states = [None for _ in range(2)]
    all_episodic_data = [[episodic_dict_generator() for _ in range(n_envs)] for _ in range(2)]
    original_actions = None
    dataset = defaultdict(structured_dataset_generator)
    eps_done = 0
    action_history = [[] for _ in range(n_envs)]
    cooldowns = np.array([decorelating_steps] * n_envs)
    reward_indices = np.zeros(shape=(n_envs, ), dtype=np.int64)
    all_infos = [None, None]
    steps_taken = 0
    with tqdm(total=n_eps, disable=suppress_progress) as pbar:
        while eps_done < n_eps:
            steps_taken += 1
            # In reality, the time index changes whenever a different action is taken
            duplicate_ep_starts, original_ep_starts = all_episode_starts
            consistent = ~deviated & ~original_ep_starts
            if not np.all(duplicate_ep_starts == original_ep_starts):
                raise ValueError(f"Env reset unexpectedly: {np.where(duplicate_ep_starts != original_ep_starts)}, steps_taken: {steps_taken}")
            duplicate_info, original_info = all_infos
            if duplicate_info is not None:
                for i in range(n_envs):
                    if not consistent[i]:
                        continue
                    for key in ["trace_idx"]:
                        if duplicate_info[i][key] != original_info[i][key]:
                            # ensure the environments are synchronized
                            raise ValueError((f"Found mismatch of {key} on idx {i}. "
                                              f"steps_taken: {original_info[i]['curr_t_idx']}."
                                              f"{duplicate_info[i][key]} != {original_info[i][key]}"))
            
            for env_idx, (env, is_duplicate, episodic_data) in enumerate(
                                                    zip(
                                                        [duplicate_env, original_env],
                                                        [True, False],
                                                        all_episodic_data
                                                        )
                                                    ):
                consistent = ~deviated | all_episode_starts[0]
                _, original_obs = all_observations
                observations = all_observations[env_idx]
                if is_duplicate:
                    observations[consistent] = original_obs[consistent]
                with th.no_grad():
                    actions, all_states[env_idx] = model.predict(
                        observations, 
                        state=all_states[env_idx], 
                        episode_start=all_episode_starts[env_idx], 
                        deterministic=True)
                    obs_tensor = th.as_tensor(observations, dtype=th.float32, device=DEVICE)
                    encoded_observations = model.policy.extract_features(obs_tensor).cpu().detach().numpy()
                
                to_record = [False for ___ in range(n_envs)]
                for action_env_idx in range(n_envs):
                    past_actions = action_history[action_env_idx]
                    if len(past_actions) >= PAST_HORIZON:
                        if np.all(np.array(past_actions[-PAST_HORIZON:]) < ACTION_THRESHOLD):
                            to_record[action_env_idx] = True
                to_record = np.array(to_record, dtype=bool)
                to_record = to_record & ~deviated
                to_record = to_record & (cooldowns == 1) & ~all_episode_starts[env_idx]

                rand_act_mask = to_record & is_duplicate
                rand_actions = action_generator.sample(actions=actions, mask=rand_act_mask)
                if original_actions is None:
                    original_actions = np.zeros_like(actions)
                original_actions[rand_actions != None] = actions[rand_actions != None]
                actions[rand_actions != None] = rand_actions[rand_actions != None]
                if not is_duplicate: # the original environment
                    deviated[to_record] = True
                    for rank, action in enumerate(actions):
                        action_history[rank].append(action)

                next_observations, rewards, dones, infos = env.step(actions)
                all_infos[env_idx] = infos

                for i in range(n_envs):
                    done = dones[i]
                    info = infos[i]
                    all_episode_starts[env_idx][i] = done
                    trace_idx = info["trace_idx"]
                    episodic_data[i]["cluster_label"] = cluster_labels[trace_idx]
                    if to_record[i]:
                        reward_indices[i] = 0
                        episodic_data[i]["observations"].append(observations[i])
                        episodic_data[i]["encoded_observations"].append(encoded_observations[i])
                        episodic_data[i]["actions"].append(actions[i])
                        episodic_data[i]["previous_sending_rates"].append(info["previous_sending_rate"])
                        episodic_data[i]["is_action_random"].append(rand_actions[i] != None)
                        episodic_data[i]["original_actions"].append(original_actions[i])
                        for key in WEIGHTS:
                            episodic_data[i][key].append(nan_array_generator((follow_n_steps_after, )))
                    if (cooldowns[i] > 1 or to_record[i]) and (len(episodic_data[i]["actions"]) > 0):
                        idx = reward_indices[i]
                        for key, weight in WEIGHTS.items():
                            episodic_data[i][key][-1][idx] = info[INFO_LOOKUP[key]] * weight
                        if not is_duplicate:
                            reward_indices[i] += 1
                    if done:
                        if not is_duplicate:
                            reward_indices[i] = 0
                            cooldowns[i] = decorelating_steps + 1
                            deviated[i] = False
                        if len(episodic_data[i]["actions"]) > 0:
                            for key in episodic_data[i]:
                                if key in ["actions", "label", "cluster_label"]:
                                    dtype = np.int32
                                elif key == "is_action_random":
                                    dtype = bool
                                else:
                                    dtype = np.float32
                                episodic_data[i][key] = np.asarray(episodic_data[i][key], dtype=dtype)
                            
                            dataset[trace_idx][env_idx].append(episodic_data[i])
                            all_episodic_data[env_idx][i] = episodic_dict_generator()
                            if not is_duplicate:
                                pbar.update(1)
                                eps_done += 1
                all_observations[env_idx] = next_observations
                if not is_duplicate:
                    cooldowns[to_record] = follow_n_steps_after
                    cooldowns[~to_record] -= 1
                    cooldowns = np.clip(cooldowns, 1, follow_n_steps_after)
    return dataset
        
def create_dataset() -> None:
    """Generate and save the standard datasets.
    """
    train_features = load_trace_features(split="train")
    train_labels = train_features.cluster_labels
    n_train_traces = train_labels.shape[0]


    val_features = load_trace_features(split="validation")
    val_labels = val_features.cluster_labels
    n_val_traces = val_labels.shape[0]
    
    train_env_kwargs = dict(sampler_kwargs=dict(split="train", sampling_func_cls="input_weighted"), 
        is_action_continuous=True)
    val_env_kwargs = dict(test=True, sampler_kwargs=dict(split="validation"),
        is_action_continuous = True)
    train_env = make_vec_env(SimulatedNetworkEnv, n_envs=N_ENVS, seed=SEED, 
        env_kwargs=train_env_kwargs, vec_env_cls=SubprocVecEnv,
        vec_env_kwargs=dict(start_method="forkserver"))
    test_env = make_vec_env(SimulatedNetworkEnv, n_envs=N_ENVS, seed=SEED, 
        env_kwargs=val_env_kwargs, vec_env_cls=SubprocVecEnv,
        vec_env_kwargs=dict(start_method="forkserver"))
    
    for rank in range(N_ENVS):
        train_idx = int((rank*n_train_traces)//N_ENVS)
        val_idx = int((rank*n_val_traces)//N_ENVS)
        train_env.env_method("set_sampler_attr", "idx", 
                train_idx, indices=rank)
        test_env.env_method("set_sampler_attr", "idx", 
                val_idx, indices=rank)
        
    model = A2C.load(MODEL_PATH)

    action_generator = ContinuousRandomActionGenerator(
        bins= CONTINUOUS_ACTION_BINS,
        possible_actions=CONTINUOUS_POSSIBLE_RAND_ACTIONS, replace_prob=0.15)
    for dataset_dir, n_eps in [[TRAIN_DATASET_DIR, N_EPS], [TEST_DATASET_DIR, N_TEST_EPS]]:
        dataset = generate_dataset(env=train_env, model=model, action_generator=action_generator,
            n_eps = n_eps, 
            cluster_labels=train_labels)
        save_dataset(dataset=dataset, dataset_dir=dataset_dir)
        
    train_env.close()
    test_env.close()
        
def create_differentiated_dataset() -> None:
    """Create the differentiated datasets (for event detection)
    """
    val_features = load_trace_features(split="validation")
    val_labels = val_features.cluster_labels
    n_val_traces = val_labels.shape[0]
    

    val_env_kwargs = dict(test=True, sampler_kwargs=dict(split="validation"),
        is_action_continuous = True)
    test_env = make_vec_env(SimulatedNetworkEnv, n_envs=N_ENVS, seed=SEED, 
        env_kwargs=val_env_kwargs, vec_env_cls=SubprocVecEnv,
        vec_env_kwargs=dict(start_method="forkserver"))
    duplicate_test_env = make_vec_env(SimulatedNetworkEnv, n_envs=N_ENVS, seed=SEED, 
        env_kwargs=val_env_kwargs, vec_env_cls=SubprocVecEnv,
        vec_env_kwargs=dict(start_method="forkserver"))
    
    for rank in range(N_ENVS):
        val_idx = int((rank*n_val_traces)//N_ENVS)
        test_env.env_method("set_sampler_attr", "idx", 
                val_idx, indices=rank)
        duplicate_test_env.env_method("set_sampler_attr", "idx", 
                val_idx, indices=rank)
        
    model = A2C.load(MODEL_PATH)

    action_generator = ContinuousRandomActionGenerator(
        bins= CONTINUOUS_ACTION_BINS,
        possible_actions=CONTINUOUS_POSSIBLE_RAND_ACTIONS, replace_prob=0.15)
    dataset= generate_differentiated_dataset(original_env=test_env, duplicate_env = duplicate_test_env, model=model, 
        action_generator= action_generator,
        n_eps = N_DIFF_EPS, 
        cluster_labels=val_labels)
    save_differentiated_dataset(dataset=dataset, dataset_dir=DIFF_DATASET_DIR)
        
    test_env.close()
    duplicate_test_env.close()
    
    
def create_datasets_for_aux_controllers() -> None:    
    """Create the differentiated datasets (for experiment with auxiliary loss training)
    """
    from auxiliary_a2c import AuxiliaryA2C
    
    val_features = load_trace_features(split="validation")
    val_labels = val_features.cluster_labels
    n_val_traces = val_labels.shape[0]
    

    val_env_kwargs = dict(test=True, sampler_kwargs=dict(split="validation"),
        is_action_continuous = True)
    test_env = make_vec_env(SimulatedNetworkEnv, n_envs=N_ENVS, seed=SEED, 
        env_kwargs=val_env_kwargs, vec_env_cls=SubprocVecEnv,
        vec_env_kwargs=dict(start_method="forkserver"))

    for rank in range(N_ENVS):
        val_idx = int((rank*n_val_traces)//N_ENVS)
        test_env.env_method("set_sampler_attr", "idx", 
                val_idx, indices=rank)


    action_generator = ContinuousRandomActionGenerator(
        bins= CONTINUOUS_ACTION_BINS,
        possible_actions=CONTINUOUS_POSSIBLE_RAND_ACTIONS, replace_prob=0.15)
    
    for reward_weight in AUXILIARY_REWARD_WEIGHTS:
        model = AuxiliaryA2C.load(AUXILIARY_MODEL_PATH.format(weight=reward_weight))
        dataset_dir = Path(AUXILIARY_DATASET_DIR.format(weight=reward_weight))
        dataset = generate_dataset(
            env=test_env, model=model, 
            action_generator=action_generator,
            n_eps =  N_TEST_EPS, 
            cluster_labels=val_labels)
        save_dataset(dataset=dataset, dataset_dir=dataset_dir)
        
    test_env.close()
    
if __name__ == "__main__":
    from argparse import ArgumentParser
    
    parser = ArgumentParser()
    parser.add_argument("--create_dataset", action="store_true", default=False, 
                        help= ("whether to create the standard datasets "
                              "based on the parameters in global_constans.py"))
    parser.add_argument("--create_diff_dataset", action="store_true", default=False,
                        help= ("whether to create the differentiated datasets "
                              "for event detection, based on the parameters "
                              "in global_constans.py"))
    parser.add_argument("--create_aux_datasets", action="store_true", default=False,
                        help= ("whether to create the evaluation datasets "
                              "for auiliary joint training experiment, "
                              "based on the parameters in global_constans.py"))
    args = parser.parse_args()
    if args.create_dataset:
        create_dataset()
    if args.create_diff_dataset:
        create_differentiated_dataset()
    if args.create_aux_datasets:
        create_datasets_for_aux_controllers()