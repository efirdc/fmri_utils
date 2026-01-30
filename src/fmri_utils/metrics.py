from __future__ import annotations

from typing import Sequence, Union, Callable, Any, Tuple

import numpy as np
from sklearn.neighbors import NearestNeighbors


def top_knn_test(
    Y,
    Y_pred,
    Y_pred_ids,
    k: Union[int, Sequence[int]],
    metric: str = "euclidean",
):
    neighbors = NearestNeighbors(metric=metric)

    if not isinstance(k, (list, tuple, np.ndarray)):
        k = [k]

    neighbors.fit(Y)

    nearest_ids = neighbors.kneighbors(Y_pred, n_neighbors=np.max(k), return_distance=False)
    Y_pred_ids = Y_pred_ids[:, None]
    accuracy = [np.any(nearest_ids[:, : int(some_k)] == Y_pred_ids, axis=1).mean() for some_k in k]
    return accuracy


def evaluation_permutation_test(
    y,
    y_pred,
    split_ids,
    evaluation_function,
    num_permutations: int = 10000,
):
    """
    Test-time permutation test for a fixed set of predictions.

    Shuffles labels *within* splits defined by `split_ids` and recomputes the evaluation metric
    to build a null distribution.

    Parameters
    ----------
    y:
        True labels (1D). Will be shuffled within split groups.
    y_pred:
        Predicted scores/labels (1D). Kept fixed across permutations.
    split_ids:
        Split/run IDs (1D), same length as y. Labels are shuffled independently within each split.
    evaluation_function:
        Callable taking (y, y_pred) and returning a scalar score (e.g., sklearn.metrics.roc_auc_score).
    num_permutations:
        Number of permutations used to build the null distribution.

    Returns
    -------
    real_score, null_mean, null_std, empirical_p, z_value
        - empirical_p is two-sided (based on absolute deviation from the null mean)
        - z_value is a signed two-sided z (positive if real_score > null_mean)
    """
    y = np.asarray(y)
    y_pred = np.asarray(y_pred)
    split_ids = np.asarray(split_ids)

    if y.shape != y_pred.shape:
        raise ValueError(f"y and y_pred must have the same shape; got {y.shape} vs {y_pred.shape}")
    if y.shape != split_ids.shape:
        raise ValueError(f"y and split_ids must have the same shape; got {y.shape} vs {split_ids.shape}")
    if y.ndim != 1:
        raise ValueError(f"Expected 1D arrays; got y.ndim={y.ndim}")
    if int(num_permutations) < 1:
        raise ValueError("num_permutations must be >= 1")

    real_score = float(evaluation_function(y, y_pred))

    rng = np.random.default_rng()
    null_scores = np.empty(int(num_permutations), dtype=float)
    uniq = np.unique(split_ids)

    y_perm = np.array(y, copy=True)
    for k in range(int(num_permutations)):
        # Shuffle within each split independently
        for sid in uniq:
            idx = np.flatnonzero(split_ids == sid)
            if idx.size <= 1:
                continue
            y_perm[idx] = y[idx[rng.permutation(idx.size)]]
        null_scores[k] = float(evaluation_function(y_perm, y_pred))

    null_mean = float(np.mean(null_scores))
    null_std = float(np.std(null_scores, ddof=1)) if null_scores.size > 1 else float("nan")

    # Two-sided empirical p-value around the null mean
    dev_real = abs(real_score - null_mean)
    dev_null = np.abs(null_scores - null_mean)
    empirical_p = (1.0 + float(np.sum(dev_null >= dev_real))) / (null_scores.size + 1.0)

    # Convert to signed two-sided z-value via normal quantile.
    # For two-sided p: z_abs = isf(p/2)
    try:
        from scipy.stats import norm
    except ImportError as e:
        raise ImportError(
            "scipy is required for z-value conversion (scipy.stats.norm.isf). "
            "Install scipy or change evaluation_permutation_test to return p-values only."
        ) from e

    p_clip = float(np.clip(empirical_p, np.finfo(float).tiny, 1.0))
    z_abs = float(norm.isf(p_clip / 2.0))
    z_value = float(np.sign(real_score - null_mean) * z_abs)

    return real_score, null_mean, null_std, empirical_p, z_value