from argparse import ArgumentParser
from pathlib import Path
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch as th
from torch import nn
from stable_baselines3.ppo.ppo import PPO
from stable_baselines3.a2c.a2c import A2C
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from train_utils import AnnealingCallBack, SharedProgressCallBack
from stable_baselines3.common.callbacks import EvalCallback

from cc_rl.network_sim import SimulatedNetworkEnv
from cc_rl.trace_loader import N_TRACES, load_trace_features
from cc_rl.sampler_utils import SharedWeightedData
from cc_rl.utils import SharedTestingData, terminate_shared_data
import numpy as np
import gymnasium as gym
import ray

if not ray.is_initialized():
    ray.init(address = "auto", log_to_driver= False)

SEEDS = [13, 103, 223, 347, 463, 607, 743, 883, 919, 937]
N_ENVS = 16
TRAIN_STEPS = int(5e6)
DEVICE = "cuda" if th.cuda.is_available() else "cpu"
LARGE_DATA_DIR = Path(__file__).parent.parent / "large_crystalbox_data"
LARGE_DATA_DIR = LARGE_DATA_DIR.resolve()
CHECKPOINT_DIR = LARGE_DATA_DIR / "reward_pred_models"
LOG_DIR = LARGE_DATA_DIR / "reward_pred_logs"

N_EVAL_EPS = N_TRACES["test"] + N_ENVS * 2

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


def main(idx: int, sampling_func_cls: str, weighted_alpha: float = None):

    seed = SEEDS[idx]

    th.manual_seed(seed)
    sampling_name = sampling_func_cls
    if sampling_func_cls == "weighted" and weighted_alpha is not None:
        sampling_name = f"{sampling_name}_{weighted_alpha:.2f}"

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

    model = A2C("MlpPolicy", env=train_env, n_steps=15, gamma=0.975, 
        ent_coef=.1, vf_coef=0.05, learning_rate=.000125, 
        use_sde=True,
        #batch_size= 25 * N_ENVS * 1,
        max_grad_norm=0.25, policy_kwargs=policy_kwargs, verbose=1, seed=seed, 
        device=DEVICE
    )
        
    model.learn(TRAIN_STEPS, callback=[ent_callback, eval_callback])
    model.save(CHECKPOINT_DIR / model_name)

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
    parser.add_argument("--idx", default=0,
        help="index of the seeding to use", type=int, required=False)

    args = parser.parse_args()
    main(idx = args.idx, sampling_func_cls = args.sampling_func, weighted_alpha=args.weighted_alpha)