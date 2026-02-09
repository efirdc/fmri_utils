from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Tuple, Dict, Any

import numpy as np
import nibabel as nib
from tqdm import tqdm

from .metrics import prepare_within_split_label_permutations, metric_permutation_test
from .decoding import reconstruct_searchlight_volume


def _as_windowwise_scores(y_scores) -> np.ndarray:
    """
    Normalize searchlight predictions to an array with leading dims (n_windows, n_samples).

    Some wrappers may return dicts; if so, we accept {'y_pred': array}.
    """
    if isinstance(y_scores, dict):
        if "y_pred" in y_scores:
            y_scores = y_scores["y_pred"]
        else:
            raise ValueError(
                f"Expected y_scores to be an array or dict with key 'y_pred'; got keys={list(y_scores.keys())}"
            )
    arr = np.asarray(y_scores)
    if arr.ndim < 2:
        raise ValueError(f"Expected y_scores to be at least 2D (n_windows, n_samples, ...); got shape={arr.shape}")
    return arr


def compute_windowwise_metric(
    y_true: np.ndarray,
    y_pred_2d,
    *,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    y_transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    permutation_test: bool = False,
    split_ids: Optional[np.ndarray] = None,
    num_permutations: int = 10000,
    seed: int = 0,
    optimized_permutation_test: Optional[Callable[..., tuple[float, float, float, float, float]]] = None,
    progress_desc: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """
    Compute a metric for each window, optionally with test-time permutation testing.

    Parameters
    ----------
    y_true:
        1D array of true labels/targets, length n_samples.
    y_pred_2d:
        2D array of predicted scores/labels with shape (n_windows, n_samples),
        or dict {'y_pred': array}.
    metric_fn:
        Callable (y_true, y_pred_1d) -> float.
    y_transform:
        Optional function applied once to y_true before computing the metric (e.g., binarization).
    permutation_test:
        If True, compute null mean/std and two-sided empirical p and signed z per window.
    split_ids:
        Required when permutation_test=True unless optimized_permutation_test provides its own scheme.
    optimized_permutation_test:
        Optional optimized permutation test implementation (e.g., roc_auc_permutation_test_fast).
        Signature must accept (y, y_pred, split_ids=..., y_permuted=..., num_permutations=..., seed=...)
        and return (real, null_mean, null_std, p, z).
    center:
        Optional center for two-sided p/z sign for the generic path.
    """
    y_true = np.asarray(y_true)
    if y_true.ndim != 1:
        raise ValueError("y_true must be 1D")
    y = y_transform(y_true) if y_transform is not None else y_true
    y = np.asarray(y)
    if y.shape != y_true.shape:
        raise ValueError("y_transform must preserve shape of y_true")

    y_pred_arr = _as_windowwise_scores(y_pred_2d)
    if y_pred_arr.shape[1] != y.shape[0]:
        raise ValueError(
            f"y_pred has n_samples={y_pred_arr.shape[1]} but y_true has n_samples={y.shape[0]}"
        )

    n_windows = int(y_pred_arr.shape[0])
    metric = np.full(n_windows, np.nan, dtype=float)

    null_mean = null_std = p_val = z_val = None
    y_permuted = None
    if bool(permutation_test):
        if split_ids is None and optimized_permutation_test is None:
            raise ValueError("permutation_test=True requires split_ids (or an optimized permutation test that doesn't).")
        if split_ids is not None:
            y_permuted = prepare_within_split_label_permutations(
                y,
                np.asarray(split_ids),
                num_permutations=int(num_permutations),
                seed=int(seed),
            )
        null_mean = np.full(n_windows, np.nan, dtype=float)
        null_std = np.full(n_windows, np.nan, dtype=float)
        p_val = np.full(n_windows, np.nan, dtype=float)
        z_val = np.full(n_windows, np.nan, dtype=float)

    it = range(n_windows)
    if bool(permutation_test):
        it = tqdm(it, total=n_windows, desc=(progress_desc or "Metric (perm-test)"), unit="window", mininterval=0.5)

    for i in it:
        scores = y_pred_arr[i]
        # Skip trivial all-constant score vectors (only meaningful for 1D scores)
        if scores.ndim == 1 and np.all(scores == scores[0]):
            continue
        metric[i] = float(metric_fn(y, scores))

        if bool(permutation_test):
            real, nm, ns, p, z = metric_permutation_test(
                y,
                scores,
                metric_fn,
                split_ids=split_ids,
                num_permutations=int(num_permutations),
                seed=int(seed),
                y_permuted=y_permuted,
                optimized_test=optimized_permutation_test,
            )
            metric[i] = real
            null_mean[i] = nm
            null_std[i] = ns
            p_val[i] = p
            z_val[i] = z

    out: Dict[str, np.ndarray] = {"metric": metric}
    if bool(permutation_test):
        out["null_mean"] = null_mean  # type: ignore[assignment]
        out["null_std"] = null_std  # type: ignore[assignment]
        out["p"] = p_val  # type: ignore[assignment]
        out["z"] = z_val  # type: ignore[assignment]
    return out


def compute_and_save_searchlight_metric_maps(
    *,
    out_dir: Path,
    subject_name: str,
    mask_vol: np.ndarray,
    searchlight_centers: np.ndarray,
    affine: np.ndarray,
    metric_name: str,
    y_true: np.ndarray,
    y_pred_2d,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    y_transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    permutation_test: bool = False,
    split_ids: Optional[np.ndarray] = None,
    num_permutations: int = 10000,
    seed: int = 0,
    optimized_permutation_test: Optional[Callable[..., tuple[float, float, float, float, float]]] = None,
    progress_desc: Optional[str] = None,
) -> Dict[str, Path]:
    """
    Compute windowwise metric (+ optional permutation maps) and save as NIfTI volumes.

    Parameters `mask_vol`, `searchlight_centers`, and `affine` define how to reconstruct
    windowwise values into a 3D NIfTI volume.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    computed = compute_windowwise_metric(
        y_true,
        y_pred_2d,
        metric_fn=metric_fn,
        y_transform=y_transform,
        permutation_test=bool(permutation_test),
        split_ids=split_ids,
        num_permutations=int(num_permutations),
        seed=int(seed),
        optimized_permutation_test=optimized_permutation_test,
        progress_desc=progress_desc,
    )

    # Map suffix -> windowwise values
    maps: Dict[str, np.ndarray] = {metric_name: computed["metric"]}
    if bool(permutation_test):
        maps[f"{metric_name}_null_mean"] = computed["null_mean"]
        maps[f"{metric_name}_null_std"] = computed["null_std"]
        maps[f"{metric_name}_p"] = computed["p"]
        maps[f"{metric_name}_z"] = computed["z"]

    outputs: Dict[str, Path] = {}
    for suffix, values in maps.items():
        img = reconstruct_searchlight_volume(
            mask_vol=mask_vol,
            searchlight_centers=searchlight_centers,
            values=np.asarray(values),
            affine=affine,
        )
        out_path = out_dir / f"{subject_name}_searchlight_{suffix}.nii.gz"
        nib.save(img, out_path.as_posix())
        outputs[suffix] = out_path

    return outputs

