from __future__ import annotations

from typing import Tuple

import numpy as np


def standardize_train_apply(
    train: np.ndarray,
    test: np.ndarray,
    *,
    eps: float = 1.0e-7,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit column standardization on train rows and apply it to test rows."""

    train_values = np.asarray(train, dtype=np.float32)
    test_values = np.asarray(test, dtype=np.float32)
    mean = np.mean(train_values, axis=0)
    scale = np.std(train_values, axis=0)
    safe_scale = np.where(scale > eps, scale, 1.0)
    train_z = (train_values - mean) / safe_scale
    test_z = (test_values - mean) / safe_scale
    zero_variance = scale <= eps
    train_z[:, zero_variance] = 0.0
    test_z[:, zero_variance] = 0.0
    return train_z, test_z, mean.astype(np.float32), safe_scale.astype(np.float32)


def residualize_run(
    values: np.ndarray,
    nuisance: np.ndarray,
    *,
    add_intercept: bool = True,
) -> np.ndarray:
    """Regress nuisance columns from one run in its original units."""

    y = np.asarray(values, dtype=np.float64)
    design = np.asarray(nuisance, dtype=np.float64)
    if y.ndim != 2 or design.ndim != 2 or y.shape[0] != design.shape[0]:
        raise ValueError("values and nuisance must be 2D with matching rows")
    if add_intercept:
        design = np.column_stack([np.ones(design.shape[0], dtype=np.float64), design])
    if design.shape[1] == 0:
        return np.asarray(y, dtype=np.float32)
    beta, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    return np.asarray(y - design @ beta, dtype=np.float32)


def voxelwise_correlation(observed: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    observed_centered = observed - np.mean(observed, axis=0, keepdims=True)
    predicted_centered = predicted - np.mean(predicted, axis=0, keepdims=True)
    numerator = np.sum(observed_centered * predicted_centered, axis=0)
    denominator = np.sqrt(
        np.sum(observed_centered ** 2, axis=0)
        * np.sum(predicted_centered ** 2, axis=0)
    )
    return np.divide(
        numerator,
        denominator,
        out=np.full(numerator.shape, np.nan, dtype=np.float64),
        where=denominator > 0,
    ).astype(np.float32)


def voxelwise_r2(observed: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    residual_ss = np.sum((observed - predicted) ** 2, axis=0)
    total_ss = np.sum((observed - np.mean(observed, axis=0)) ** 2, axis=0)
    ratio = np.divide(
        residual_ss,
        total_ss,
        out=np.full(total_ss.shape, np.nan, dtype=np.float64),
        where=total_ss > 0,
    )
    return (1.0 - ratio).astype(np.float32)


def weighted_fisher_mean(correlations: np.ndarray, row_counts: np.ndarray) -> np.ndarray:
    values = np.asarray(correlations, dtype=np.float64)
    weights = np.maximum(np.asarray(row_counts, dtype=np.float64) - 3.0, 1.0)
    finite = np.isfinite(values)
    z_values = np.arctanh(np.clip(values, -0.999999, 0.999999))
    numerator = np.nansum(np.where(finite, z_values * weights[:, None], 0.0), axis=0)
    denominator = np.sum(np.where(finite, weights[:, None], 0.0), axis=0)
    mean_z = np.divide(
        numerator,
        denominator,
        out=np.full(numerator.shape, np.nan),
        where=denominator > 0,
    )
    return np.tanh(mean_z).astype(np.float32)
