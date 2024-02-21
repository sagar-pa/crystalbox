from pathlib import Path
import numpy as np
import torch as th

#CrsytalBox Dataset Generation parameters
DEVICE = "cuda" if th.cuda.is_available() else "cpu"
N_ENVS = 12
N_EPS = 45000
SEED = 12
N_ACTIONS = 1
N_TEST_EPS = 6000
N_DIFF_EPS = 500
WEIGHTS = {
    "throughputs": 10,
    "latencies": -1e3,
    "losses": -2e3
}
NORMALIZATION_WEIGHTS = {
    "throughputs": 1./10,
    "latencies": -1./1e3,
    "losses": -1./2e3
}
INFO_LOOKUP = {
    "throughputs": "throughput",
    "latencies": "latency",
    "losses": "loss"
}

LARGE_DATA_DIR = Path(__file__).parent.parent / "large_crystalbox_data"
LARGE_DATA_DIR = LARGE_DATA_DIR.resolve()

MODEL_PATH = LARGE_DATA_DIR / "reward_pred_models" / "random__0.zip"
AUXILIARY_REWARD_WEIGHTS = [1, 10, 50]
AUXILIARY_DATASET_DIR = LARGE_DATA_DIR / "reward_dataset" / "aux_{weight}_test"
AUXILIARY_MODEL_PATH = LARGE_DATA_DIR / "reward_pred_models" / "random_auxiliary_{weight}.00__0"
CONTINUOUS_ACTION_BINS = np.array([-1, -0.25, 0, 0.25, 1])
CONTINUOUS_POSSIBLE_RAND_ACTIONS = [[0., 1.], # 0, -25
                        [0., 1.], # 1, -5
                        [-1., 1.], # 2, 0
                        [0., -1.], # 3, +5
                        [0., -1.], # 4, +25
                        ]

TRAIN_DATASET_DIR = LARGE_DATA_DIR / "reward_dataset" / "train"
TEST_DATASET_DIR = LARGE_DATA_DIR / "reward_dataset" / "test"
DIFF_DATASET_DIR = LARGE_DATA_DIR / "reward_dataset" / "test_diff"

#CrystalBox Training parameters
CRYSTALBOX_SAVE_PATH = Path(__file__).parent / "models"
CRYSTALBOX_DATA_SAVE_PATH = Path(__file__).parent / "data"

NORMALIZE_REWARDS = True
N_STEP= 5
GAMMA = 0.95
SHUFFLE_SEED = 1
N_ACTIONS = 1
MAX_VALS = {
    "throughputs": 500,
    "latencies": 5,
    "losses": 1.0 
}
TRAIN_SEED = 36
BATCH_SIZE = 50
N_EPOCHS = 120
LOSS_WEIGHTS = {"throughputs": 0.25, 
    "latencies": 0.5, 
    "losses": 0.25}