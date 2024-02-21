from typing import NamedTuple
import numpy as np
import pandas as pd
from pathlib import Path
import json
from cc_rl.trace_utils import determine_n_clusters
from gymnasium.utils import seeding


RANGES = {
    "train": {
        "br": [100., 500.],
        "lat": [.05, .3],
        "queue": [2., 50.],
        "lr": [0., .02]
    },
    "validation": {
        "br": [100., 500.],
        "lat": [.05, .3],
        "queue": [2., 50.],
        "lr": [0., .02]
    },
    "test": {
        "br": [50., 1000.],
        "lat": [.025, .5],
        "queue": [2., 75.],
        "lr": [0., .03]
    }
}
DATASET_SEEDS = {
    "train": 13,
    "validation": 12,
    "test": 11
}

N_TRACES = {
    "train": 5000,
    "validation": 4000,
    "test": 3000
}


def make_traces(split: str) -> np.ndarray:
    n_traces = N_TRACES[split]
    trace_range = RANGES[split]
    rand, _ = seeding.np_random(seed=DATASET_SEEDS[split])
    traces = np.zeros(shape=(n_traces, 4), dtype=np.float64)
    br_samples = np.geomspace(*trace_range["br"], num=n_traces)
    lat_samples = np.linspace(*trace_range["lat"], num=n_traces)
    queue_samples = np.geomspace(*trace_range["queue"], num=n_traces)
    queue_samples = np.array(queue_samples, dtype=np.int64)
    eps = 1e-3
    lr_samples = np.geomspace(trace_range["lr"][0] + eps,
                                trace_range["lr"][1] + eps,
                                num=n_traces
                            )
    lr_samples -= eps
    rand.shuffle(br_samples)
    rand.shuffle(lat_samples)
    rand.shuffle(queue_samples)
    rand.shuffle(lr_samples)
    for i, (br, lat, queue, lr) in enumerate(zip(
                                                br_samples, lat_samples, 
                                                queue_samples, lr_samples
                                                )):
        traces[i] = [br, lat, queue, lr]
    return traces


class TraceFeatures(NamedTuple):
    features: np.ndarray
    cluster_labels: np.ndarray
    cluster_dists: np.ndarray


def extract_and_save_trace_features(dataset_name: str = "default", split: str = "train") -> None:
    """
    Extract and save the filtered trace features to trace_data/settings/trace_features/{dataset_name}.json
        Filters the features and clusters the data automatically, saving the cluster labels 
    
    Args:
        dataset_name: The name of the dataset to feed to generate_dataset
            (see toy_abr_gym/trace_generator/generate_dataset)
    """
    traces = make_traces(split)
    trace_features = pd.DataFrame(traces, columns=["br", "lat", "queue", "lr"])
    _, cluster_labels, dist, __ = determine_n_clusters(trace_features, return_labels=True)

    trace_features_lst = trace_features.to_dict("records")
    to_save = {}
    for i, trace in enumerate(traces):
        to_save[i] = dict(cluster_label=int(cluster_labels[i]),
                            cluster_dist=dist[i].tolist(),
                            **trace_features_lst[i])

    with open(Path(__file__).parent / "trace_data" / "trace_features" / f"{dataset_name}_{split}.json", "w") as f:
        json.dump(to_save, f, indent=2, sort_keys=True)

def load_trace_features(dataset_name: str = "default", split: str = "train") -> TraceFeatures:
    """
    Load the saved features for the given dataset name. 
    Assumes extrace_and_save_trace_features was already called.

    Args:
        dataset_name: The name (one of ["majority_slow", "balanced", "majority_fast"])
    Returns:
        The trace features, a np.ndarray of shape [n_traces, n_features]
        The cluster labels, a np.ndarray of shape [n_traces, ]
        The cluster dists, a np.ndarray of shape [n_traces, n_clusters]
    """
    with open(Path(__file__).parent / "trace_data" / "trace_features" / f"{dataset_name}_{split}.json", "r") as f:
        trace_features = json.load(f)
    features = [None] * len(trace_features)
    cluster_labels = [None] * len(trace_features)
    cluster_dists = [None] * len(trace_features)
    for i, trace_feature in trace_features.items():
        feature = {k: trace_feature[k] for k in sorted(trace_feature.keys()) if k not in [
                                                                "cluster_label", 
                                                                "trace_label", 
                                                                "cluster_dist"]}
        i = int(i)
        features[i] = list([feature[key] for key in sorted(feature.keys())])
        cluster_labels[i] = int(trace_feature["cluster_label"])
        cluster_dists[i] = list(trace_feature["cluster_dist"])

    features = np.array(features, dtype=np.float64)
    cluster_labels = np.array(cluster_labels, dtype = np.int64)
    cluster_dists = np.array(cluster_dists, dtype = np.float64)

    return TraceFeatures(features, cluster_labels, cluster_dists)
    