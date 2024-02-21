import numpy as np
import pandas as pd
from cc_rl.trace_loader import (
    load_trace_features,
    make_traces,
)
from typing import Tuple
from cc_rl.trace_utils import get_dist_weights
from cc_rl.sampler_utils import add_sample, get_probabilities, N_STEP
import ray


class Sampler:
    def __init__(self, 
            np_random: np.random, 
            sampling_func_cls: str = "random",
            split: str = "train",
            shared_weighted_data_name: str = None,
            shared_testing_data_name: str = None
            ) -> None:

        self.np_random = np_random
        self.sampling_func_cls = sampling_func_cls
        if split.lower() not in ["train", "validation", "test"]:
            raise ValueError("Split Must be either ['train', 'validation', 'test']")
        self.all_traces = make_traces(split=split.lower())
        if shared_weighted_data_name is None:
            self.shared_weighted_data = None
        else:
            self.shared_weighted_data = ray.get_actor(shared_weighted_data_name, namespace="weighted")
        if shared_testing_data_name is None:
            self.shared_testing_data = None
        else:
            self.shared_testing_data = ray.get_actor(shared_testing_data_name, namespace="test")
        self.current_trace_rewards = []
        self.probs = None
        self.is_init = True
        self.split = split
        self.idx = -1
        self.setup_probs()

    def setup_probs(self):
        if self.sampling_func_cls == "random" and self.is_init:
            self.is_init = False

        if self.sampling_func_cls == "input_weighted" and self.is_init:
            trace_features = load_trace_features(split=self.split)
            dist = trace_features.cluster_dists
            self.probs = get_dist_weights(dist)
            self.is_init = False

        if self.sampling_func_cls == "weighted" and self.is_init:
            if self.shared_weighted_data is None:
                raise ValueError(("Attempted to use weighted without passing"
                                " in a shared sampling actor"))

            from sklearn.preprocessing import scale

            trace_features = load_trace_features(split=self.split)
            self.trace_features = scale(trace_features.features)
            self.cluster_dists = trace_features.cluster_dists
            self.cluster_labels = trace_features.cluster_labels
            self.is_init = False

    def record_action_reward(self, action: int, reward: float,
            throughput: float = None, latency: float=None,
            loss: float = None) -> None:
        if self.sampling_func_cls != "weighted":
            return
        else:
            self.current_trace_rewards.append(reward)

    def record_episode(self, idx: int) -> None:
        """
        Log the reward data observed during the episode (for weighted sampling)
        """
        if self.sampling_func_cls == "weighted":
            if len(self.current_trace_rewards) >= N_STEP:
                trace_feature = self.trace_features[idx]
                cluster_label = self.cluster_labels[idx]
                rewards = np.array(self.current_trace_rewards, dtype=np.float64)
                add_sample(shared_weighted_data=self.shared_weighted_data, 
                    trace_feature=trace_feature, rewards=rewards, cluster=cluster_label)
            self.current_trace_rewards = []

    def sample(self, epsilon: float = 0.1) -> Tuple[np.ndarray, int]:
        if self.np_random.random() < epsilon:
            p = None
        else:
            if self.sampling_func_cls == "weighted":
                p = get_probabilities(shared_weighted_data = self.shared_weighted_data)
            else:
                p = self.probs
        
        idx = self.np_random.choice(len(self.all_traces), p=p)
        self.idx = idx
        return self.all_traces[idx], self.idx

    def iterative_sample(self) -> Tuple[np.ndarray, int]:
        if self.shared_testing_data is None:
            self.idx += 1
            if not 0 <= self.idx < len(self.all_traces):
                self.idx = 0
        else:
            self.idx = ray.get(self.shared_testing_data.get_index_to_test.remote())
        return self.all_traces[self.idx], self.idx