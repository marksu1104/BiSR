from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import numpy as np


def dist_mat(query_vecs):
    """
    Calculate cosine sim
    :param query_vecs:
    :return:
    """
    sim = np.matmul(query_vecs, np.transpose(query_vecs))
    return 1 - sim


def cluster_entities(entity_vecs, t=0.7):
    """
    Return cluster assignment of entity vectors.
    :param entity_vecs:
    :param t: desired threshold for hierarchical clustering
    :return:
    """
    dists = dist_mat(entity_vecs.cpu().numpy())
    np.fill_diagonal(dists, 0)
    dists = np.clip(dists, 0, None)
    # Ensure symmetry (handle floating point precision issues)
    dists = (dists + dists.T) / 2
    # Handle NaN/Inf values - replace with max finite value
    dists = np.nan_to_num(dists, nan=1.0, posinf=1.0, neginf=0.0)
    # Convert to condensed form (upper triangle) to avoid squareform validation issues
    n = dists.shape[0]
    condensed = dists[np.triu_indices(n, k=1)]
    # build tree
    zavg = linkage(condensed, method='average')
    c = fcluster(zavg, criterion='distance', t=t)

    return c
