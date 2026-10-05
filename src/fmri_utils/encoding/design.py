"""Encoding designs built on the training runs only: PCA, extra columns, FIR delays.

``prepare_design`` is the classic naturalistic-stimulus design (Huth/LeBel style):

1. z-score the features, then PCA (randomized, seed 2023);
2. append any extra columns (a word rate, say) after the PCA, z-scored, so a single regressor
   is not one of hundreds of PCA inputs smeared across components;
3. FIR delays per run (zero-padded, nothing crosses a run boundary);
4. z-score the delayed design.

Every learned step is fitted on the training runs and applied unchanged to the test run.
``design_from_blocks`` takes a design as it is (lag 0), z-scored with the training statistics:
for predictors that are already on the response clock, such as other subjects' responses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class Design:
    train: np.ndarray          # training rows (all training runs stacked) x design columns
    test: np.ndarray           # test rows x design columns
    run_groups: np.ndarray     # training run index per training row
    pca_components: int


def delayed(matrix: np.ndarray, delays: Sequence[int]) -> np.ndarray:
    """Zero-padded delayed copies of a run's rows, side by side, one block per delay (>= 1)."""
    matrix = np.asarray(matrix, dtype=np.float32)
    blocks = []
    for delay in delays:
        block = np.zeros_like(matrix)
        if delay == 0:
            block = matrix.copy()
        else:
            block[delay:] = matrix[:-delay]
        blocks.append(block)
    return np.hstack(blocks)


def _zscore_stats(train: np.ndarray):
    mean = train.mean(axis=0, dtype=np.float64)
    std = train.std(axis=0, dtype=np.float64)
    std[std < 1e-8] = 1.0
    return mean, std


def prepare_design(train_features: Sequence[np.ndarray], test_features: np.ndarray, *, pca_components: int = 256,
                   delays: Sequence[int] = (1, 2, 3, 4), extra_train: Sequence[np.ndarray] | None = None,
                   extra_test: np.ndarray | None = None, seed: int = 2023) -> Design:
    """z-score, PCA, append extras, FIR-delay, z-score (all fitted on the training runs)."""
    from sklearn.decomposition import PCA
    raw = np.vstack([np.asarray(m, dtype=np.float32) for m in train_features])
    mean, std = _zscore_stats(raw)
    stacked = ((raw - mean) / std).astype(np.float32)
    test = ((np.asarray(test_features, dtype=np.float32) - mean) / std).astype(np.float32)
    n_components = min(int(pca_components), stacked.shape[0] - 1, stacked.shape[1])
    if n_components < 2:
        raise ValueError(f"design too small for PCA: {stacked.shape}")
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=seed).fit(stacked)
    reduced, start = [], 0
    for matrix in train_features:
        stop = start + matrix.shape[0]
        reduced.append(pca.transform(stacked[start:stop]).astype(np.float32))
        start = stop
    test_reduced = pca.transform(test).astype(np.float32)
    if extra_train is not None:
        if extra_test is None or len(extra_train) != len(reduced):
            raise ValueError("extra_train needs one matrix per training run, and extra_test")
        extra_mean, extra_std = _zscore_stats(np.vstack([np.asarray(m, dtype=np.float32) for m in extra_train]))

        def append(block, extra):
            extra = np.asarray(extra, dtype=np.float32).reshape(block.shape[0], -1)
            return np.hstack([block, ((extra - extra_mean) / extra_std).astype(np.float32)])
        reduced = [append(r, e) for r, e in zip(reduced, extra_train)]
        test_reduced = append(test_reduced, extra_test)
    train_delayed = [delayed(m, delays) for m in reduced]
    train = np.vstack(train_delayed)
    groups = np.concatenate([np.full(m.shape[0], i, dtype=np.int16) for i, m in enumerate(train_delayed)])
    mean, std = _zscore_stats(train)
    return Design(train=((train - mean) / std).astype(np.float32),
                  test=((delayed(test_reduced, delays) - mean) / std).astype(np.float32),
                  run_groups=groups, pca_components=n_components)


def design_from_blocks(train_blocks: Sequence[np.ndarray], test_block: np.ndarray) -> Design:
    """A design used as is (lag 0), z-scored with the training statistics."""
    raw = np.vstack(train_blocks)
    mean, sd = raw.mean(axis=0), raw.std(axis=0)
    sd[sd < 1e-8] = 1.0
    groups = np.concatenate([np.full(b.shape[0], i, dtype=np.int16) for i, b in enumerate(train_blocks)])
    return Design(train=((raw - mean) / sd).astype(np.float32), test=((test_block - mean) / sd).astype(np.float32),
                  run_groups=groups, pca_components=raw.shape[1])
