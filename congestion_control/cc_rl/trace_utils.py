import numpy as np
import pandas as pd
from typing import Union, Tuple, Callable


def cluster(features: np.ndarray, n_clusters: int, max_starts: int = 300
        ) -> Tuple[Callable, np.ndarray, np.ndarray, float]:
    """
    Cluster the given features into n_clusters max_starts number of times, 
        and maximize the mean log expectation of the Gaussian Mixture Model
    
    Args:
        features: The features to cluster
        n_clusters: The number of clusters (mixture components to have)
        max_starts: The max number of random states to try
    Returns:
        A tuple of (The gmm model, 
            The predicted labels (array of shape (n_samples, )),
            The predicted scores of each sampler for each component (array of shape (n_samples, n_components)),
            The mean log expectation)
    """
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import scale

    scaled_features = scale(features, with_mean=True)
    scores = []

    for i in range(max_starts):
        gmm = GaussianMixture(random_state=i, 
            n_components=n_clusters, max_iter=500, n_init=1, init_params="k-means++")
        labels = gmm.fit_predict(scaled_features)
        score = gmm.score(scaled_features)
        scores.append([gmm, labels, gmm.predict_proba(scaled_features), score])

    scores.sort(key= lambda x:x[-1])
    return scores[-1]

def determine_n_clusters(features: pd.DataFrame,
        min_clusters: int = 4,
        max_clusters: int = 9, 
        min_cluster_size: float = 0.000001, 
        return_labels: bool = False,) -> Union[int, Tuple[int, np.ndarray]]:
    """
    Automatically determine the number of clusters 
        between min and max clusters using mahalanobis-distance silhoutte score as the metric
    
    Args:
        features: The features returned by tsfesh of the traces
        min_clusters: The minimum number of clusters to try
        max_clusters: The maximum number of clusters to try
        min_cluster_size: If the smallest cluster is smaller than 
            this fraction (proportional to the number of traces), this clustering is ignored.
        return_labels: Whether or not to return the ideal cluster labels found
    Returns:
        The ideal number of clusters, if return_labels is False
        (number of clusters, labels), otherwise 
    """    
    
    from sklearn.preprocessing import scale
    from sklearn.metrics import silhouette_score
    
    scaled_features = scale(features.values, with_mean=True)

    scores = []
    min_cluster_size = min_cluster_size* features.shape[0]
    for n_clusters in range(min_clusters, max_clusters+1):
        _, labels, dist, ___ = cluster(scaled_features, n_clusters=n_clusters)
        _, counts = np.unique(labels, return_counts=True)
        score = silhouette_score(scaled_features, labels, metric="euclidean")
        if np.amin(counts) <= min_cluster_size:
            score = -1
        scores.append([n_clusters, labels, dist, score])
        
    sorted_scores = sorted(scores, key= lambda x: x[-1])
    best_n_clusters, best_labels, dist, _ = sorted_scores[-1]
    
    if return_labels:
        return (best_n_clusters, best_labels, dist, scores)
    else:
        return best_n_clusters

def get_dist_weights(dist: np.ndarray, class_weights: np.ndarray = None,
        return_class_weights: bool = False, max_pool_inflation: float = None) -> np.ndarray:
    """
    The distribution equivalent of get_cluster_weights. 
    Works by assuming every trace has a probability of being in every cluster, 
        and adjusting them to be at class_weights.

    Args:
        dist: 2-D array of shape [n_samples, n_classes] 
            describing the prob to each sampling belonging to each class
        class_weights: 1-D array of shape [n_classes, ] 
            describing the target class weights, if not given, equal weights are assumed.
        return_class_weights: whether or not to return the true class weights used
        max_pool_inflation: if given, is the multiplicative factor 
            by which the weight change will be clipped to.

    Returns:
        The calculated weights of shape (labels) and
            class weights of shape (unique labels) if return_class_weights is True
    """
    class_proportions = dist.sum(axis=0)
    class_proportions = class_proportions / class_proportions.sum()
    cluster_pools = class_weights
    if cluster_pools is None:
        cluster_pools = np.ones((dist.shape[1], ), dtype=np.float64) / dist.shape[1]
    cluster_pools /= cluster_pools.sum()
    if max_pool_inflation is not None:
        cluster_pools = np.clip(
            cluster_pools,
            class_proportions / max_pool_inflation,
            class_proportions * max_pool_inflation)
    cluster_pools /= cluster_pools.sum()

    adjustment = cluster_pools / class_proportions
    weights = dist @ adjustment

    if return_class_weights:
        return (weights / weights.sum(), cluster_pools)
    else:
        return weights / weights.sum()
