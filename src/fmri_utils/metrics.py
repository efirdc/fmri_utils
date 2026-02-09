from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Union, Callable, Optional

import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.stats import norm
import scipy.stats as st


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
    evaluation_function,
    *,
    split_ids=None,
    num_permutations: int = 10000,
    seed: int = 0,
    y_permuted: Optional[np.ndarray] = None,
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
    seed:
        RNG seed for reproducibility when generating permutations internally.
    y_permuted:
        Optional precomputed permuted labels array of shape (num_permutations, n_samples).
        If provided, `split_ids/seed` are ignored for permutation generation.
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

    if y.shape != y_pred.shape:
        raise ValueError(f"y and y_pred must have the same shape; got {y.shape} vs {y_pred.shape}")
    if y.ndim != 1:
        raise ValueError(f"Expected 1D arrays; got y.ndim={y.ndim}")
    if int(num_permutations) < 1:
        raise ValueError("num_permutations must be >= 1")

    real_score = float(evaluation_function(y, y_pred))

    n_perm = int(num_permutations)
    if y_permuted is None:
        if split_ids is None:
            raise ValueError("Provide split_ids or y_permuted for permutation generation.")
        y_permuted = prepare_within_split_label_permutations(y, split_ids, num_permutations=n_perm, seed=int(seed))
    y_permuted = np.asarray(y_permuted)
    if y_permuted.shape != (n_perm, y.shape[0]):
        raise ValueError(
            f"y_permuted must have shape (num_permutations, n_samples)=({n_perm}, {y.shape[0]}), got {y_permuted.shape}"
        )

    null_scores = np.empty(n_perm, dtype=float)
    for k in range(n_perm):
        null_scores[k] = float(evaluation_function(y_permuted[k], y_pred))

    null_mean = float(np.mean(null_scores))
    null_std = float(np.std(null_scores, ddof=1)) if null_scores.size > 1 else float("nan")

    # Two-sided empirical p-value around the null mean
    dev_real = abs(real_score - null_mean)
    dev_null = np.abs(null_scores - null_mean)
    empirical_p = (1.0 + float(np.sum(dev_null >= dev_real))) / (null_scores.size + 1.0)

    # Convert to signed two-sided z-value via normal quantile.
    # For two-sided p: z_abs = isf(p/2)
    p_clip = float(np.clip(empirical_p, np.finfo(float).tiny, 1.0))
    z_abs = float(norm.isf(p_clip / 2.0))
    z_value = float(np.sign(real_score - null_mean) * z_abs)

    return real_score, null_mean, null_std, empirical_p, z_value


def metric_permutation_test(
    y,
    y_pred,
    evaluation_function: Callable[[np.ndarray, np.ndarray], float],
    *,
    split_ids=None,
    num_permutations: int = 10000,
    seed: int = 0,
    y_permuted: Optional[np.ndarray] = None,
    optimized_test: Optional[
        Callable[..., tuple[float, float, float, float, float]]
    ] = None,
) -> tuple[float, float, float, float, float]:
    """
    Unified test-time permutation test entry point.

    If `optimized_test` is provided, it will be used (e.g., `roc_auc_permutation_test_fast`).
    Otherwise falls back to the generic `evaluation_permutation_test` implementation.

    The optimized implementation must return:
        (real_score, null_mean, null_std, p_two_sided, z_signed)

    """
    if optimized_test is not None:
        return optimized_test(
            y,
            y_pred,
            split_ids=split_ids,
            y_permuted=y_permuted,
            num_permutations=int(num_permutations),
            seed=int(seed),
        )
    return evaluation_permutation_test(
        y,
        y_pred,
        evaluation_function,
        split_ids=split_ids,
        num_permutations=int(num_permutations),
        seed=int(seed),
        y_permuted=y_permuted,
    )


def prepare_within_split_label_permutations(
    y: np.ndarray,
    split_ids: np.ndarray,
    *,
    num_permutations: int,
    seed: int = 0,
) -> np.ndarray:
    """
    Pre-generate permuted labels by shuffling *within* split groups.

    Returns
    -------
    y_permuted : np.ndarray
        Array of shape (num_permutations, n_samples) with permuted labels.
    """
    y = np.asarray(y)
    split_ids = np.asarray(split_ids)
    if y.ndim != 1 or split_ids.ndim != 1:
        raise ValueError("y and split_ids must be 1D")
    if y.shape[0] != split_ids.shape[0]:
        raise ValueError("y and split_ids must have the same length")
    n_perm = int(num_permutations)
    if n_perm < 1:
        raise ValueError("num_permutations must be >= 1")

    rng = np.random.default_rng(int(seed))
    y_permuted = np.empty((n_perm, y.shape[0]), dtype=y.dtype)
    uniq = np.unique(split_ids)
    for sid in uniq:
        idx = np.flatnonzero(split_ids == sid)
        if idx.size == 0:
            continue
        y_split = y[idx]
        if idx.size <= 1:
            y_permuted[:, idx] = y_split
            continue
        for b in range(n_perm):
            y_permuted[b, idx] = rng.permutation(y_split)
    return y_permuted


def roc_auc_from_ranks(pos_mask: np.ndarray, ranks: np.ndarray) -> float:
    """
    AUC via Mann–Whitney U / rank-sum, given precomputed ranks (average ties).
    """
    pos_mask = np.asarray(pos_mask, dtype=bool)
    ranks = np.asarray(ranks, dtype=float)
    n_pos = int(np.sum(pos_mask))
    n_neg = int(pos_mask.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    R_pos = float(np.sum(ranks[pos_mask]))
    U = R_pos - (n_pos * (n_pos + 1) / 2.0)
    return float(U / (n_pos * n_neg))


def roc_auc_permutation_test_fast(
    y_bin: np.ndarray,
    scores: np.ndarray,
    *,
    split_ids: Optional[np.ndarray] = None,
    y_permuted: Optional[np.ndarray] = None,
    num_permutations: int = 10000,
    seed: int = 0,
) -> tuple[float, float, float, float, float]:
    """
    Fast test-time permutation test for ROC AUC using pre-ranked scores.

    This is optimized for the common decoding case:
    - fixed prediction scores per window
    - many permutations of labels within split/run

    Returns
    -------
    auc_obs, null_mean, null_std, p_two_sided, z_signed
        p-value is two-sided around the null mean (computed from permutations).
    """
    y_bin = np.asarray(y_bin).astype(int)
    scores = np.asarray(scores, dtype=float)
    if y_bin.ndim != 1 or scores.ndim != 1:
        raise ValueError("y_bin and scores must be 1D")
    if y_bin.shape[0] != scores.shape[0]:
        raise ValueError("y_bin and scores must have the same length")

    n_perm = int(num_permutations)
    if y_permuted is None:
        if split_ids is None:
            raise ValueError("Provide split_ids or y_permuted for permutation generation.")
        y_permuted = prepare_within_split_label_permutations(y_bin, split_ids, num_permutations=n_perm, seed=int(seed))
    y_permuted = np.asarray(y_permuted)
    if y_permuted.shape != (n_perm, y_bin.shape[0]):
        raise ValueError(
            f"y_permuted must have shape (num_permutations, n_samples)=({n_perm}, {y_bin.shape[0]}), got {y_permuted.shape}"
        )

    ranks = st.rankdata(scores, method="average")
    auc_obs = roc_auc_from_ranks(y_bin.astype(bool), ranks)

    n_pos = int(np.sum(y_bin == 1))
    n_neg = int(np.sum(y_bin == 0))
    if n_pos == 0 or n_neg == 0:
        return float("nan"), float("nan"), float("nan"), float("nan"), float("nan")

    # Build permuted positive masks as bool for fast dot with ranks.
    pos_perm = (y_permuted.astype(int) == 1)
    R_pos_perm = (pos_perm @ ranks.astype(float)).astype(float)
    U_perm = R_pos_perm - (n_pos * (n_pos + 1) / 2.0)
    auc_perm = U_perm / float(n_pos * n_neg)

    null_mean = float(np.mean(auc_perm))
    null_std = float(np.std(auc_perm, ddof=1)) if auc_perm.size > 1 else float("nan")

    dev_obs = abs(auc_obs - null_mean)
    dev_perm = np.abs(auc_perm - null_mean)
    p = (1.0 + float(np.sum(dev_perm >= dev_obs))) / (auc_perm.size + 1.0)

    z_abs = float(norm.isf(np.clip(p, np.finfo(float).tiny, 1.0) / 2.0))
    z = float(np.sign(auc_obs - null_mean) * z_abs)
    return float(auc_obs), null_mean, null_std, float(p), z