import numpy as np
from torch.nn.functional import mse_loss
from torch.utils.data import DataLoader
import torch as th
from tqdm.auto import tqdm
from typing import Callable
import pickle as pk
from auxiliary_a2c import DecomposedReturnsObjective
from pathlib import Path
from argparse import ArgumentParser


from global_constants import (
    CRYSTALBOX_DATA_SAVE_PATH, BATCH_SIZE, INFO_LOOKUP, DEVICE, 
    AUXILIARY_MODEL_PATH, AUXILIARY_DATASET_DIR)
from crystalbox_model import (
    RewardDataset, N_SAMPLES)

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

def test(dataloader: DataLoader, model: Callable, device: str, split: str):
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
                pred_key = INFO_LOOKUP[key]
                samples = pred[pred_key].sample((N_SAMPLES, ))
                preds[key].extend(pred[pred_key].mean.squeeze(1).detach().cpu().tolist())
                mse = mse_loss(samples, data[key].unsqueeze(0).repeat(N_SAMPLES, 1, 1), reduction="none").mean(dim=0)
                mse_losses[key].extend(mse.detach().cpu().tolist())


    for key in mse_losses:
        mse = np.round(np.mean(mse_losses[key]), 6)
        tqdm.write(f"{key:25} MSE: {mse}", end="\n")

    return mse_losses, preds

def evaluate_shared_loss_strat(predictor_weight: float) -> None: 

    model = DecomposedReturnsObjective(
                keys_to_predict= AUX_KEYS_TO_PREDICT,
                scales = AUX_SCALES,
                maximums= AUX_MAXIMUMS,
                n_step= AUX_N_STEP,
                gamma = AUX_GAMMA,
                feature_dim= AUX_FEATURE_DIM,
                action_dim= AUX_ACTION_DIM,
                shared_net_arch= AUX_SHARED_NET_ARCH,
                predictor_net_arch= AUX_PREDICTOR_NET_ARCH,
                loss_weights= AUX_LOSS_WEIGHTS
            )
    model.load_state_dict(th.load(Path(AUXILIARY_MODEL_PATH.format(weight=predictor_weight))))
    model = model.to(DEVICE)
    model = model.eval()
    
    test_dataset = RewardDataset(Path(AUXILIARY_DATASET_DIR), "test")
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    test_mse, preds = test(test_dataloader, model, DEVICE, "Test")
    save_path = CRYSTALBOX_DATA_SAVE_PATH / f"regression_whitebox_aux_{predictor_weight}_test_mses.pk"
    with open(save_path, "wb") as f:
        pk.dump(dict(mses=test_mse, preds=preds), f)
        
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--auxiliary_coef", type=float,
                        help=("The weight on the shared auxiliary model. "
                              "Must point to an already trained model "
                              "and auxiliary dataset"), required=True)
    args = parser.parse_args()
    evaluate_shared_loss_strat(predictor_weight=args.auxiliary_coef)