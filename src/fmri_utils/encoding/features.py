from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd


FEATURE_METADATA_COLUMNS = {
    "subject_id",
    "task",
    "run",
    "run_id",
    "tr_index",
    "tr_onset_s",
    "semantic_valid",
    "valid",
}


def load_feature_matrix(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load a numeric feature matrix and its optional row-validity vector."""

    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".npy"):
        values = np.load(path, allow_pickle=False)
        valid = np.ones(values.shape[0], dtype=bool)
    elif suffixes.endswith(".npz"):
        with np.load(path, allow_pickle=False) as payload:
            if "features" in payload.files:
                key = "features"
            else:
                candidates = [
                    name
                    for name in payload.files
                    if name not in {"valid", "semantic_valid"}
                    and np.asarray(payload[name]).ndim == 2
                ]
                if len(candidates) != 1:
                    raise ValueError(
                        "NPZ must contain a 'features' array or exactly one 2D array: {}"
                        .format(path)
                    )
                key = candidates[0]
            values = payload[key]
            valid_key = "valid" if "valid" in payload.files else "semantic_valid"
            valid = (
                np.asarray(payload[valid_key], dtype=bool)
                if valid_key in payload.files
                else np.ones(values.shape[0], dtype=bool)
            )
    elif suffixes.endswith(".csv") or suffixes.endswith(".tsv"):
        sep = "\t" if suffixes.endswith(".tsv") else ","
        frame = pd.read_csv(path, sep=sep)
        validity_name = next(
            (name for name in ("semantic_valid", "valid") if name in frame.columns),
            None,
        )
        valid = (
            frame[validity_name].fillna(False).astype(bool).to_numpy()
            if validity_name
            else np.ones(len(frame), dtype=bool)
        )
        numeric = frame.select_dtypes(include=[np.number, "bool"]).drop(
            columns=[name for name in FEATURE_METADATA_COLUMNS if name in frame.columns],
            errors="ignore",
        )
        if numeric.shape[1] == 0:
            raise ValueError("Feature table has no numeric feature columns: {}".format(path))
        values = numeric.to_numpy()
    else:
        raise ValueError("Unsupported feature format: {}".format(path))

    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("Feature matrix must be 2D; got {} from {}".format(values.shape, path))
    if valid.shape != (values.shape[0],):
        raise ValueError("Feature validity vector does not match the feature rows")
    if not np.all(np.isfinite(values)):
        raise ValueError("Feature matrix contains non-finite values: {}".format(path))
    return values, valid


def build_lagged_features(
    features: np.ndarray,
    lags: Tuple[int, ...],
    *,
    mode: str = "concat",
    source_valid: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Shift features so row ``t - lag`` predicts BOLD row ``t``."""

    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("features must be 2D")
    n_rows, n_features = values.shape
    valid_source = (
        np.ones(n_rows, dtype=bool)
        if source_valid is None
        else np.asarray(source_valid, dtype=bool)
    )
    if valid_source.shape != (n_rows,):
        raise ValueError("source_valid must have one value per feature row")

    blocks = []
    valid = np.ones(n_rows, dtype=bool)
    for lag in lags:
        if lag < 0:
            raise ValueError("lags must be non-negative")
        shifted = np.zeros((n_rows, n_features), dtype=np.float32)
        lag_valid = np.zeros(n_rows, dtype=bool)
        if lag == 0:
            shifted[:] = values
            lag_valid[:] = valid_source
        elif lag < n_rows:
            shifted[lag:] = values[:-lag]
            lag_valid[lag:] = valid_source[:-lag]
        blocks.append(shifted)
        valid &= lag_valid

    if mode == "concat":
        output = np.concatenate(blocks, axis=1)
    elif mode == "mean":
        output = np.mean(np.stack(blocks, axis=0), axis=0, dtype=np.float32)
    else:
        raise ValueError("mode must be 'concat' or 'mean'")
    return output.astype(np.float32, copy=False), valid


def block_permute_rows(
    values: np.ndarray,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Permute contiguous row blocks while preserving rows within each block."""

    n_rows = values.shape[0]
    blocks = [np.arange(start, min(start + block_length, n_rows)) for start in range(0, n_rows, block_length)]
    order = rng.permutation(len(blocks))
    indices = np.concatenate([blocks[index] for index in order])
    return np.asarray(values[indices], dtype=np.float32)
