"""The noise ceiling of a repeated stimulus, and correlations normalised by it."""

from __future__ import annotations

import numpy as np


def noise_ceiling(repeats: np.ndarray) -> np.ndarray:
    """The correlation ceiling of the repeat mean (repeats x times x voxels).

    Schoppe et al.'s signal/noise decomposition: signal variance = (n var(mean) - mean var) / (n - 1),
    floored at 0; the ceiling is sqrt(signal / var(mean)), in [0, 1]."""
    values = np.asarray(repeats, dtype=np.float64)
    if values.ndim != 3 or values.shape[0] < 2:
        raise ValueError("repeats must be (n_repeats, n_times, n_voxels)")
    n = values.shape[0]
    total = np.var(values.mean(axis=0), axis=0, ddof=1)
    within = np.mean(np.var(values, axis=1, ddof=1), axis=0)
    signal = np.maximum((n * total - within) / (n - 1), 0.0)
    ratio = np.divide(signal, total, out=np.zeros_like(signal), where=total > 0)
    return np.sqrt(ratio).clip(0.0, 1.0).astype(np.float32)


def normalise(correlation: np.ndarray, ceiling: np.ndarray, floor: float = 0.3) -> np.ndarray:
    """r / max(ceiling, floor): the floor keeps unreliable voxels from blowing up."""
    if floor <= 0:
        raise ValueError("the floor must be positive")
    return (np.asarray(correlation, dtype=np.float32) / np.maximum(np.asarray(ceiling, dtype=np.float32), floor)).astype(np.float32)
