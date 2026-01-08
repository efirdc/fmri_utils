import warnings

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectorMixin
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from typing import Dict, Optional, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from scipy.interpolate import griddata
import nibabel as nib


def decoding(
    X: np.ndarray,
    y: np.ndarray,
    split_ids: np.ndarray,
    selector: Optional[SelectorMixin] = None,
    model_class: BaseEstimator = Ridge,
    model_params: Optional[Dict] = None,
    permutation_test: bool = False,
    permutation_iterations: int = 1000,
    scale_X=True,
    normalize_y_pred_fold=False,
):
    if model_params is None:
        model_params = {}

    y = np.asarray(y)
    unique_split_ids = np.unique(split_ids)

    if permutation_test:
        y_pred = np.full((*y.shape, int(permutation_iterations)), np.nan, dtype=float)
    else:
        y_pred = np.full(y.shape, np.nan, dtype=float)

    models = []
    num_regressors_selected: Optional[list] = [] if selector is not None else None
    # "regressors" == model features (columns of X)
    num_regressors = int(X.shape[1]) if X.ndim >= 2 else 0
    for test_split_id in unique_split_ids:
        train_mask = split_ids != test_split_id
        test_mask = ~train_mask

        X_train = X[train_mask]
        y_train = y[train_mask]
        X_test = X[test_mask]

        if scale_X:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        if selector is not None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                feature_mask = selector.fit(X_train, y_train).get_support()
            X_train = X_train[:, feature_mask]
            X_test = X_test[:, feature_mask]
            if num_regressors_selected is not None:
                num_regressors_selected.append(int(np.sum(feature_mask)))
            # print(f'Filtered out {np.sum(~feature_mask)} voxels, {np.sum(feature_mask)} voxels remaining')
        elif num_regressors_selected is not None:
            # selector was passed but resulted in None (shouldn't happen); keep shape consistent
            num_regressors_selected.append(int(X_train.shape[1]))

        model = model_class(**model_params)
        if permutation_test:
            y_pred_permutations = []
            for _ in range(permutation_iterations):
                y_train_shuffled = np.copy(y_train)
                np.random.shuffle(y_train_shuffled)
                model.fit(X_train, y_train_shuffled)
                y_pred_shuffled = np.asarray(model.predict(X_test))
                # If y is (N,1), some models may return (N,) predictions; normalize to (N,1)
                if y.ndim == 2 and y.shape[1] == 1 and y_pred_shuffled.ndim == 1:
                    y_pred_shuffled = y_pred_shuffled[:, None]
                if normalize_y_pred_fold:
                    y_pred_shuffled = (y_pred_shuffled - np.mean(y_pred_shuffled)) / np.std(y_pred_shuffled)
                y_pred_permutations.append(y_pred_shuffled)
            y_pred_permutations = np.stack(y_pred_permutations, axis=-1)
            y_pred[test_mask, ...] = y_pred_permutations
        else:
            model.fit(X_train, y_train)
            y_pred_fold = np.asarray(model.predict(X_test))
            if y.ndim == 2 and y.shape[1] == 1 and y_pred_fold.ndim == 1:
                y_pred_fold = y_pred_fold[:, None]
            if normalize_y_pred_fold:
                y_pred_fold = (y_pred_fold - np.mean(y_pred_fold)) / np.std(y_pred_fold)
            y_pred[test_mask, ...] = y_pred_fold
            models.append(model)

    return {
        'y_pred': y_pred,
        'models': models,
        'num_regressors': num_regressors,
        'num_regressors_selected': num_regressors_selected,
    }


def searchlight_decoding(
    X: np.ndarray,
    X_windows: Sequence[np.ndarray],
    y: np.ndarray,
    split_ids: np.ndarray,
    n_jobs: int = 1,
    tqdm_mininterval: float = 0.1,
    tqdm_miniters: Optional[int] = None,
    **kwargs,
):
    # Validate shapes
    num_voxels, num_stimuli = X.shape
    if y.shape[0] != num_stimuli:
        raise ValueError('y must have length equal to the number of stimuli (second dimension of X)')
    if split_ids.shape[0] != num_stimuli:
        raise ValueError('run_ids must have length equal to the number of stimuli (second dimension of X)')
    num_windows = len(X_windows)
    if num_windows == 0:
        warnings.warn('searchlight_decoding called with no windows; returning empty predictions')
        return np.empty((0, *y.shape), dtype=float)
    # Warn on empty windows
    empty_windows = sum(1 for w in X_windows if np.asarray(w).size == 0)
    if empty_windows > 0:
        warnings.warn(f'{empty_windows} of {num_windows} windows are empty; predictions for these entries will be NaN')

    # Always disable permutation_test for searchlight output shape consistency
    kwargs_no_perm = {**kwargs, 'permutation_test': False}

    # Collect predictions: one row per window, one column per stimulus
    y_pred_all = np.full((num_windows, *y.shape), np.nan, dtype=float)
    # NOTE: we no longer support returning model attributes here; `decoding` returns a dict including models.

    def _process_window(idx: int):
        voxel_indices = np.asarray(X_windows[idx], dtype=int)
        if voxel_indices.size == 0:
            # Indicate empty; caller will fill NaNs
            return idx, None
        X_sub = X[voxel_indices, :].T
        ret = decoding(X_sub, y, split_ids, **kwargs_no_perm)
        return idx, ret

    tqdm_kwargs = {
        'mininterval': float(tqdm_mininterval),
    }
    if tqdm_miniters is not None:
        tqdm_kwargs['miniters'] = int(tqdm_miniters)

    if int(n_jobs) > 1:
        y_pred_rows = [None] * num_windows
        with ThreadPoolExecutor(max_workers=int(n_jobs)) as ex:
            futures = [ex.submit(_process_window, i) for i in range(num_windows)]
            for fut in tqdm(as_completed(futures), total=len(futures), desc='Searchlight decoding', unit='window', **tqdm_kwargs):
                i, ret = fut.result()
                if ret is None:
                    y_pred_rows[i] = np.full(y.shape, np.nan, dtype=float)
                else:
                    y_pred_rows[i] = ret['y_pred']
        for i in range(num_windows):
            y_pred_all[i] = y_pred_rows[i]  # type: ignore[index]
        return {'y_pred': y_pred_all, 'models': None, 'num_regressors': int(X.shape[0]), 'num_regressors_selected': None}
    else:
        for i in tqdm(range(num_windows), total=num_windows, desc='Searchlight decoding', unit='window', **tqdm_kwargs):
            voxel_indices = np.asarray(X_windows[i], dtype=int)
            if voxel_indices.size == 0:
                y_pred_all[i] = np.full(y.shape, np.nan, dtype=float)
                continue
            else:
                X_sub = X[voxel_indices, :].T
            ret = decoding(X_sub, y, split_ids, **kwargs_no_perm)
            y_pred_all[i] = ret['y_pred']

    return {'y_pred': y_pred_all, 'models': None, 'num_regressors': int(X.shape[0]), 'num_regressors_selected': None}


def get_searchlight_windows(
    mask_vol: np.ndarray,
    radius: int,
    stride: int = 1,
):
    # mask_vol: boolean volume indicating valid voxels (intersection of brain and ROI)
    if radius < 0 or stride <= 0:
        raise ValueError('radius must be >= 0 and stride must be > 0')
    mask_vol = mask_vol.astype(bool)
    sx, sy, sz = mask_vol.shape
    # Map from 3D voxel to flattened X index (masked order)
    idx_vol = np.full(mask_vol.shape, -1, dtype=np.int64)
    idx_vol[mask_vol] = np.arange(int(np.sum(mask_vol)), dtype=np.int64)

    centers = []
    windows = []
    for i in range(0, sx, stride):
        for j in range(0, sy, stride):
            for k in range(0, sz, stride):
                if not mask_vol[i, j, k]:
                    continue
                i0 = max(0, i - radius)
                i1 = min(sx - 1, i + radius)
                j0 = max(0, j - radius)
                j1 = min(sy - 1, j + radius)
                k0 = max(0, k - radius)
                k1 = min(sz - 1, k + radius)
                sub = idx_vol[i0:i1 + 1, j0:j1 + 1, k0:k1 + 1]
                inds = sub[sub >= 0].reshape(-1)
                if inds.size == 0:
                    continue
                centers.append((i, j, k))
                windows.append(inds)
    searchlight_centers = np.array(centers, dtype=int) if len(centers) > 0 else np.zeros((0, 3), dtype=int)
    return windows, searchlight_centers


def reconstruct_searchlight_volume(
    mask_vol: np.ndarray,
    searchlight_centers: np.ndarray,
    values: np.ndarray,
    affine: np.ndarray,
    fill_value: float = np.nan,
) -> nib.Nifti1Image:
    mask_vol = mask_vol.astype(bool)
    sx, sy, sz = mask_vol.shape
    # Full grid of voxel centers inside mask
    idxs = np.argwhere(mask_vol)
    # Interpolate only inside mask
    vals = griddata(
        points=np.asarray(searchlight_centers, dtype=float),
        values=np.asarray(values, dtype=float),
        xi=idxs.astype(float),
        method='linear',
        fill_value=fill_value,
    )
    vol = np.full(mask_vol.shape, fill_value, dtype=np.float32)
    vol[mask_vol] = vals.astype(np.float32)
    return nib.Nifti1Image(vol, affine)