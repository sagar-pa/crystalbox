from torch.nn.functional import mse_loss
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch as th
from tqdm.auto import tqdm
import numpy as np
from typing import Callable, Dict, Tuple, List
import pickle as pk
from argparse import ArgumentParser

from global_constants import (
    CRYSTALBOX_SAVE_PATH, CRYSTALBOX_DATA_SAVE_PATH, TRAIN_SEED, BATCH_SIZE, 
    TRAIN_DATASET_DIR, NORMALIZE_REWARDS, N_EPOCHS, LOSS_WEIGHTS, 
    TEST_DATASET_DIR, DIFF_DATASET_DIR, DEVICE)
from crystalbox_model import (
    RewardDataset, RewardNetwork, init_weights, N_SAMPLES, calculate_runtimes)



def test(dataloader: DataLoader, model: Callable, device: str, split: str
         ) -> Tuple[Dict[str, List[float]], Dict[str, List[float]]]:
    """Test the predictor, writings logs to tqdm while doing it

    Args:
        dataloader: the dataloader to load test batches from
        model: the model to evaluate
        device: the device (cpu/cuda) to evaluate on
        split: the split of the dataloader. Only used for printing.

    Returns:
        The MSE and Predictions, given as a dictionary of all reward components and list
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
                samples = pred[key].sample((N_SAMPLES, ))
                preds[key].extend(pred[key].mean.squeeze(1).detach().cpu().tolist())
                mse = mse_loss(samples, data[key].unsqueeze(0).repeat(N_SAMPLES, 1, 1), reduction="none").mean(dim=0)
                mse_losses[key].extend(mse.detach().cpu().tolist())


    for key in mse_losses:
        mse = np.round(np.mean(mse_losses[key]), 6)
            #mse_losses[key][label] = mse
        tqdm.write(f"{key:25} MSE: {mse}", end="\n")

    return mse_losses, preds


def train_predictor() -> None:
    th.manual_seed(TRAIN_SEED)
    g = th.Generator()
    g.manual_seed(TRAIN_SEED)

    train_dataset = RewardDataset(TRAIN_DATASET_DIR, "train", normalize_rewards=NORMALIZE_REWARDS)
    val_dataset = RewardDataset(TRAIN_DATASET_DIR, "validation", normalize_rewards=NORMALIZE_REWARDS)
    weights = train_dataset.calculate_weights(key="is_action_random", value=1, target_weight=0.4)
    weights = th.Tensor(weights)
    sampler = WeightedRandomSampler(weights, num_samples=weights.shape[0], replacement=True, generator=g)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, generator=g, sampler=sampler)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, generator=g)

    model = RewardNetwork().to(device=DEVICE)
    model = model.apply(init_weights)
    optimizer = th.optim.Adam([{"params": model.shared_net.parameters()},
                            {"params": model.throughput_predictor.parameters(), "lr": 9e-6},
                            {"params": model.latency_predictor.parameters(), "lr": 9e-6},
                            {"params": model.loss_predictor.parameters(), "lr": 9e-6},], lr=9e-6)
    n_batches = len(train_dataloader)

    with tqdm(total=N_EPOCHS, desc="epochs", position=0, leave=True) as epoch_pbar:
        for epoch in range(N_EPOCHS):
            with tqdm(total=n_batches, desc="batches", position=1, leave=False) as batch_pbar:
                tqdm.write(f"Epoch {epoch+1}\n-------------------------------", end="\n")
                for data in train_dataloader:
                    for key in data:
                        data[key] = data[key].to(DEVICE)
                    pred = model(data)
                    loss = None
                    for key in LOSS_WEIGHTS:
                        facet_loss = -th.mean(pred[key].log_prob(data[key]))
                        if loss is None:
                            loss = LOSS_WEIGHTS[key] * facet_loss
                        else:
                            loss += LOSS_WEIGHTS[key] * facet_loss

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    batch_pbar.update()
                    
            test(val_dataloader, model, DEVICE, "Validation")
            epoch_pbar.update()
    th.save(model.state_dict(), 
            CRYSTALBOX_SAVE_PATH / "reward_regression_net_.15_noise_no_overlap.pt")

def evaluate_predictor() -> None:
    model = RewardNetwork()
    model.load_state_dict(th.load(CRYSTALBOX_SAVE_PATH / "reward_regression_net_.15_noise_no_overlap.pt"))
    model = model.to(DEVICE)
    model = model.eval()

    # Evaluate on test data
    test_dataset = RewardDataset(TEST_DATASET_DIR, "test")
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_mse, preds = test(test_dataloader, model, DEVICE, "Test")
    save_path = CRYSTALBOX_DATA_SAVE_PATH / "regression_test_mses.pk"
    with open(save_path, "wb") as f:
        pk.dump(dict(mses=test_mse, preds=preds), f)
    cluster_labels = []
    is_action_random = {
        "throughputs": [],
        "latencies": [],
        "losses": []
    }
    # Save extra test data
    for batch in test_dataloader:
        cluster_labels.extend(batch["cluster_labels"].flatten().detach().cpu().tolist())
        for key in ["throughputs", "latencies", "losses"]:
            is_action_random[key].extend(batch["is_action_random"].detach().cpu().tolist())
    labels_save_path = CRYSTALBOX_DATA_SAVE_PATH / "test_dataset_cluster_labels.pk"

    with open(labels_save_path, "wb") as f:
        pk.dump(dict(cluster_labels=cluster_labels, is_action_random=is_action_random), f)


    #Evaluate on Differentiated data (for event detection)
    filtered_test_dataset = RewardDataset(DIFF_DATASET_DIR, "test")
    filtered_test_dataloader = DataLoader(filtered_test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_mse, preds = test(filtered_test_dataloader, model, DEVICE, "Test")
    save_path = CRYSTALBOX_DATA_SAVE_PATH / "regression_filtered_test_mses.pk"
    with open(save_path, "wb") as f:
        pk.dump(dict(mses=test_mse, preds=preds), f)
    modified_test_dataset = RewardDataset(DIFF_DATASET_DIR, "test", differentiated=True)
    modified_test_dataloader = DataLoader(modified_test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_mse, preds = test(modified_test_dataloader, model, DEVICE, "Test")
    save_path = CRYSTALBOX_DATA_SAVE_PATH / "regression_modified_test_mses.pk"
    with open(save_path, "wb") as f:
        pk.dump(dict(mses=test_mse, preds=preds), f)
    # Save extra test data
    original_true = {
        "throughputs": [],
        "latencies": [],
        "losses": [],
    }
    modified_true = {
        "throughputs": [],
        "latencies": [],
        "losses": [],
    }
    original_actions = []
    actions = []
    for original_batch, modified_batch in zip(filtered_test_dataloader, modified_test_dataloader):
        for key in original_true:
            original_true[key].extend(original_batch[key].squeeze(1).detach().cpu().tolist())
            modified_true[key].extend(modified_batch[key].squeeze(1).detach().cpu().tolist())
        actions.extend(original_batch["actions"].squeeze(1).detach().cpu().tolist())
        original_actions.extend(original_batch["original_actions"].detach().cpu().tolist())

    original_save_path = CRYSTALBOX_DATA_SAVE_PATH / "differentiated_dataset_original_true.pk"
    modified_save_path = CRYSTALBOX_DATA_SAVE_PATH / "differentiated_dataset_modified_true.pk"

    with open(original_save_path, "wb") as f:
        pk.dump(dict(preds=original_true, actions=actions, original_actions=original_actions), f)
    with open(modified_save_path, "wb") as f:
        pk.dump(dict(preds=modified_true, actions=actions, original_actions=original_actions), f)

    #also save runtime benchmak of the model
    save_path = CRYSTALBOX_DATA_SAVE_PATH / "cc_learned_runtimes.pk"
    run_times = calculate_runtimes(dataset= test_dataset, algorithm=model, device=DEVICE)
    with open(save_path, "wb") as f:
        pk.dump(dict(run_times=run_times), f)
        
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--train", action="store_true", default=False,
                        help="Whether to train the CrystalBox model")
    parser.add_argument("--test", action="store_true", default=False,
                        help="Whether to test the model on the dataset")
    args = parser.parse_args()
    if args.train:
        train_predictor()
    if args.test:
        evaluate_predictor()