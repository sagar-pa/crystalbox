from argparse import ArgumentParser
from pathlib import Path
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym
import torch as th
from torch import nn
from auxiliary_a2c import AuxiliaryA2C, DecomposedReturnsObjective
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from train_utils import AnnealingCallBack, SharedProgressCallBack
from stable_baselines3.common.callbacks import EvalCallback

from cc_rl.network_sim import SimulatedNetworkEnv
from cc_rl.trace_loader import N_TRACES, load_trace_features
from cc_rl.sampler_utils import SharedWeightedData
from cc_rl.utils import SharedTestingData, terminate_shared_data

import numpy as np

import ray

if not ray.is_initialized():
    ray.init(address = "auto", log_to_driver= False)
    
    
LARGE_DATA_DIR = Path(__file__).parent.parent / "large_crystalbox_data"
LARGE_DATA_DIR = LARGE_DATA_DIR.resolve()
    
CHECKPOINT_DIR = LARGE_DATA_DIR / "reward_pred_models"
LOG_DIR = LARGE_DATA_DIR / "reward_pred_logs"

SEEDS = [13, 103, 223, 347, 463, 607, 743, 883, 919, 937]
N_ENVS = 16
TRAIN_STEPS = int(7e6)
DEVICE = "cuda" if th.cuda.is_available() else "cpu"


N_EVAL_EPS = N_TRACES["test"] + N_ENVS * 2

AUX_KEYS_TO_PREDICT = ["throughput", "latency", "loss"]
AUX_SCALES = {
    "throughput": 1/10,
    "latency": -(1/1000),
    "loss": -(1/2000)
}
AUX_MAXIMUMS = {
    "throughput": 500,
    "latency": 5,
    "loss": 1.0 
}
AUX_LOSS_WEIGHTS = {
    "throughput": 0.3,
    "latency": 0.3,
    "loss": 0.4,
}
AUX_N_STEP = 5
AUX_GAMMA = 0.95
AUX_FEATURE_DIM = 128
AUX_ACTION_DIM = 1
AUX_SHARED_NET_ARCH = [128, 128]
AUX_PREDICTOR_NET_ARCH = [64]


class CongestionLayer(BaseFeaturesExtractor):

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        n_input = np.prod(observation_space.shape)
        self.cnn = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_input, features_dim),
            nn.Tanh(),
        )

        
    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.cnn(observations)

def main(idx: int, sampling_func_cls: str, weighted_alpha: float = None, 
        auxiliary_coef: float = None):

    seed = SEEDS[idx]

    th.manual_seed(seed)
    sampling_name = sampling_func_cls
    if sampling_func_cls == "weighted" and weighted_alpha is not None:
        sampling_name = f"{sampling_name}_{weighted_alpha:.2f}"
    if auxiliary_coef is not None:
        sampling_name = f"{sampling_name}_auxiliary_{auxiliary_coef:.2f}"

    train_sampler_kwargs = dict(sampling_func_cls=sampling_func_cls)

    model_name = f"{sampling_name}__{idx}"
    log_dir = LOG_DIR / model_name

    if sampling_func_cls == "weighted":
        sampling_log_dir = log_dir / "sampling"
        trace_features = load_trace_features(split="train")
        SharedWeightedData.options(lifetime="detached", 
            name=model_name, namespace="weighted").remote(
                alpha = weighted_alpha, 
                trace_features = trace_features.features,
                trace_dists = trace_features.cluster_dists, 
                log_dir = sampling_log_dir
        )
        train_sampler_kwargs["shared_weighted_data_name"] = model_name

    train_env_kwargs = dict(sampler_kwargs = train_sampler_kwargs)
    SharedTestingData.options(lifetime="detached", name = model_name, 
        namespace="test").remote(
        N_TRACES["test"]
    )
    test_env_kwargs = dict(test=True, sampler_kwargs =dict(split="test"), 
        log_dir=log_dir, use_log = True, shared_testing_data_name = model_name)


    train_env = make_vec_env(SimulatedNetworkEnv, n_envs=N_ENVS, seed=seed, 
        env_kwargs=train_env_kwargs, vec_env_cls=SubprocVecEnv, 
        vec_env_kwargs=dict(start_method="forkserver"))
    test_env = make_vec_env(SimulatedNetworkEnv, n_envs=N_ENVS, seed=seed, 
        env_kwargs=test_env_kwargs, vec_env_cls=SubprocVecEnv, 
        vec_env_kwargs=dict(start_method="forkserver"))
    

    policy_kwargs = dict(net_arch=[dict(pi=[128], vf=[128])],
                            activation_fn=nn.Tanh,
                            features_extractor_class=CongestionLayer,
                            log_std_init=-2, ortho_init=False)
    ent_callback = AnnealingCallBack("ent_coef", start=.1, end=.005, 
        total_train_steps=TRAIN_STEPS, n_train_envs=N_ENVS)
    shared_progress_updater = SharedProgressCallBack(actor_name = model_name)
    eval_freq = int((TRAIN_STEPS / N_ENVS) / 20) # about every 5% training 
    eval_callback = EvalCallback(eval_env=test_env, 
        n_eval_episodes=N_EVAL_EPS, eval_freq=eval_freq,
        callback_after_eval = shared_progress_updater,
        best_model_save_path = str(CHECKPOINT_DIR/ f"best__{model_name}"))
    auxiliary_objective = None
    if auxiliary_coef is not None:
        auxiliary_objective = DecomposedReturnsObjective(
            keys_to_predict= AUX_KEYS_TO_PREDICT,
            scales = AUX_SCALES,
            maximums= AUX_MAXIMUMS,
            n_step= AUX_N_STEP,
            gamma = AUX_GAMMA,
            feature_dim= AUX_FEATURE_DIM,
            action_dim= AUX_ACTION_DIM,
            shared_net_arch= AUX_SHARED_NET_ARCH,
            predictor_net_arch= AUX_PREDICTOR_NET_ARCH,
            loss_weights= AUX_LOSS_WEIGHTS,
        )
        auxiliary_objective.to(device=DEVICE)

    model = AuxiliaryA2C(policy="AuxiliaryCnnPolicy", env=train_env, n_steps=15, gamma=0.975, 
        ent_coef=.1, vf_coef=0.05, learning_rate=.000125, 
        use_sde=True,
        auxiliary_objective=auxiliary_objective,
        auxiliary_coef=auxiliary_coef,
        #batch_size= 25 * N_ENVS * 1,
        max_grad_norm=0.25, policy_kwargs=policy_kwargs, verbose=1, seed=seed, 
        device=DEVICE
    )
        
    model.learn(TRAIN_STEPS, callback=[ent_callback, eval_callback])
    model.save(CHECKPOINT_DIR / model_name)
    if auxiliary_objective is not None:
        th.save(auxiliary_objective.state_dict(), CHECKPOINT_DIR / f"{model_name}_predictor.pt")

    actors_to_terminate = [(model_name, "test")]
    if sampling_func_cls == "weighted":
        actors_to_terminate.append((model_name, "weighted"))
    terminate_shared_data(actors_to_terminate)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--sampling_func", 
        help="The sampling funtion to use", type=str, required=True)
    parser.add_argument("--weighted_alpha", type=float,
        help="Alpha of sampling to use", default=None, required=False)
    parser.add_argument("--auxiliary_coef", type=float, required=False,
        default= None)
    parser.add_argument("--idx", default=0,
        help="index of the seeding to use", type=int, required=False)

    args = parser.parse_args()
    main(idx = args.idx, sampling_func_cls = args.sampling_func, weighted_alpha=args.weighted_alpha, auxiliary_coef=args.auxiliary_coef)