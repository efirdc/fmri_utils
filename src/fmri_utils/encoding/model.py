from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import norm
from sklearn.linear_model import Ridge

from .config import EncodingConfig
from .data import EncodingRun, SubjectData
from .features import block_permute_rows
from .preprocessing import (
    standardize_train_apply,
    voxelwise_correlation,
    voxelwise_r2,
    weighted_fisher_mean,
)
from .splits import (
    CVPlan,
    InnerFold,
    Selection,
    build_cv_plan,
    selection_metadata,
    validate_cv_plan,
)


@dataclass
class EncodingResult:
    subject_id: str
    run_ids: Tuple[str, ...]
    fold_ids: Tuple[str, ...]
    mean_correlation: np.ndarray
    mean_r2: np.ndarray
    outer_fold_correlation: np.ndarray
    outer_fold_r2: np.ndarray
    outer_selected_alpha: np.ndarray
    outer_selected_pca_components: np.ndarray
    mean_selected_alpha: np.ndarray
    mode_selected_pca_components: np.ndarray
    observed_by_fold: Tuple[np.ndarray, ...]
    predicted_by_fold: Tuple[np.ndarray, ...]
    test_run_indices_by_fold: Tuple[np.ndarray, ...]
    test_source_rows_by_fold: Tuple[np.ndarray, ...]
    null_mean_fisher_z: np.ndarray
    null_std_fisher_z: np.ndarray
    null_standardized_effect_fisher_z: np.ndarray
    empirical_p: np.ndarray
    signed_z: np.ndarray
    metadata: Dict[str, object]


@dataclass
class _ObservedFit:
    mean_correlation: np.ndarray
    mean_r2: np.ndarray
    outer_correlation: np.ndarray
    outer_r2: np.ndarray
    outer_alpha: np.ndarray
    outer_pca: np.ndarray
    observed: Tuple[np.ndarray, ...]
    predicted: Tuple[np.ndarray, ...]
    test_run_indices: Tuple[np.ndarray, ...]
    test_source_rows: Tuple[np.ndarray, ...]


def _stack(runs: Sequence[EncodingRun], selection: Selection, field: str) -> np.ndarray:
    arrays = [
        getattr(run, field)[rows]
        for run, rows in zip(runs, selection)
        if rows.size
    ]
    if not arrays:
        raise ValueError("Cannot stack an empty row selection")
    return np.concatenate(arrays, axis=0)


def _test_row_provenance(
    runs: Sequence[EncodingRun], selection: Selection
) -> Tuple[np.ndarray, np.ndarray]:
    run_indices = []
    source_rows = []
    for run_index, (run, rows) in enumerate(zip(runs, selection)):
        if rows.size:
            run_indices.append(np.full(rows.size, run_index, dtype=np.int16))
            source_rows.append(np.asarray(run.source_rows[rows], dtype=np.int32))
    return np.concatenate(run_indices), np.concatenate(source_rows)


def _pca_train_apply(
    train: np.ndarray,
    test: np.ndarray,
    n_components: Optional[int],
) -> Tuple[np.ndarray, np.ndarray]:
    if n_components is None:
        return train, test
    maximum = min(train.shape[0] - 1, train.shape[1])
    if n_components > maximum:
        raise ValueError(
            "pca_components={} exceeds the fold maximum {}".format(n_components, maximum)
        )
    _, _, vt = np.linalg.svd(train.astype(np.float64, copy=False), full_matrices=False)
    components = vt[:n_components].T
    return (
        np.asarray(train @ components, dtype=np.float32),
        np.asarray(test @ components, dtype=np.float32),
    )


def _fit_predict_ridge(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    alpha: float,
) -> np.ndarray:
    model = Ridge(alpha=float(alpha), fit_intercept=False, solver="svd")
    model.fit(x_train, y_train)
    prediction = model.predict(x_test)
    if prediction.ndim == 1:
        prediction = prediction[:, None]
    return np.asarray(prediction, dtype=np.float32)


def _grid(config: EncodingConfig) -> List[Tuple[Optional[int], float]]:
    return [
        (components, float(alpha))
        for components in config.pca_components
        for alpha in config.ridge_alphas
    ]


def _select_grid_per_voxel(
    runs: Sequence[EncodingRun],
    inner_folds: Sequence[InnerFold],
    config: EncodingConfig,
) -> np.ndarray:
    candidates = _grid(config)
    fold_scores = []
    fold_weights = []
    for fold in inner_folds:
        x_train = _stack(runs, fold.train, "features")
        y_train = _stack(runs, fold.train, "bold")
        x_validation = _stack(runs, fold.validation, "features")
        y_validation = _stack(runs, fold.validation, "bold")
        x_train_z, x_validation_z, _, _ = standardize_train_apply(x_train, x_validation)
        y_train_z, y_validation_z, _, _ = standardize_train_apply(y_train, y_validation)

        component_cache = {}
        scores = np.full((len(candidates), y_train.shape[1]), np.nan, dtype=np.float32)
        for candidate_index, (components, alpha) in enumerate(candidates):
            if components not in component_cache:
                component_cache[components] = _pca_train_apply(
                    x_train_z, x_validation_z, components
                )
            x_fit, x_predict = component_cache[components]
            prediction = _fit_predict_ridge(x_fit, y_train_z, x_predict, alpha)
            scores[candidate_index] = voxelwise_correlation(y_validation_z, prediction)
        fold_scores.append(scores)
        fold_weights.append(max(y_validation.shape[0] - 3, 1))

    values = np.stack(fold_scores, axis=0).astype(np.float64)
    if config.fisher_z_selection:
        values = np.arctanh(np.clip(values, -0.999999, 0.999999))
    weights = np.asarray(fold_weights, dtype=np.float64)[:, None, None]
    finite = np.isfinite(values)
    numerator = np.sum(np.where(finite, values * weights, 0.0), axis=0)
    denominator = np.sum(np.where(finite, weights, 0.0), axis=0)
    mean_scores = np.divide(
        numerator,
        denominator,
        out=np.full(numerator.shape, -np.inf),
        where=denominator > 0,
    )
    return np.argmax(mean_scores, axis=0).astype(np.int32)


def _fit_observed(
    runs: Sequence[EncodingRun], config: EncodingConfig, cv_plan: CVPlan
) -> _ObservedFit:
    n_folds = len(cv_plan.outer_folds)
    n_voxels = runs[0].bold.shape[1]
    candidates = _grid(config)
    outer_corr = np.full((n_folds, n_voxels), np.nan, dtype=np.float32)
    outer_r2 = np.full_like(outer_corr, np.nan)
    outer_alpha = np.full_like(outer_corr, np.nan)
    outer_pca = np.full_like(outer_corr, np.nan)
    predictions = []
    observed = []
    test_run_indices = []
    test_source_rows = []
    row_counts = []

    for fold_index, fold in enumerate(cv_plan.outer_folds):
        selected = _select_grid_per_voxel(runs, fold.inner_folds, config)
        x_train = _stack(runs, fold.train, "features")
        y_train = _stack(runs, fold.train, "bold")
        x_test = _stack(runs, fold.test, "features")
        y_test = _stack(runs, fold.test, "bold")
        x_train_z, x_test_z, _, _ = standardize_train_apply(x_train, x_test)
        y_train_z, y_test_z, y_mean, y_scale = standardize_train_apply(y_train, y_test)
        prediction_z = np.zeros_like(y_test_z, dtype=np.float32)

        component_cache = {}
        for candidate_index, (components, alpha) in enumerate(candidates):
            target_mask = selected == candidate_index
            if not np.any(target_mask):
                continue
            if components not in component_cache:
                component_cache[components] = _pca_train_apply(x_train_z, x_test_z, components)
            x_fit, x_predict = component_cache[components]
            candidate_prediction = _fit_predict_ridge(
                x_fit, y_train_z, x_predict, alpha
            )
            prediction_z[:, target_mask] = candidate_prediction[:, target_mask]
            outer_alpha[fold_index, target_mask] = alpha
            outer_pca[fold_index, target_mask] = 0 if components is None else components

        outer_corr[fold_index] = voxelwise_correlation(y_test_z, prediction_z)
        outer_r2[fold_index] = voxelwise_r2(y_test_z, prediction_z)
        predictions.append(prediction_z * y_scale[None, :] + y_mean[None, :])
        observed.append(np.asarray(y_test, dtype=np.float32))
        run_indices, source_rows = _test_row_provenance(runs, fold.test)
        test_run_indices.append(run_indices)
        test_source_rows.append(source_rows)
        row_counts.append(y_test.shape[0])

    row_counts_array = np.asarray(row_counts, dtype=np.int32)
    mean_corr = weighted_fisher_mean(outer_corr, row_counts_array)
    finite_r2 = np.isfinite(outer_r2)
    r2_weights = row_counts_array.astype(np.float64)[:, None]
    mean_r2 = np.divide(
        np.sum(np.where(finite_r2, outer_r2 * r2_weights, 0.0), axis=0),
        np.sum(np.where(finite_r2, r2_weights, 0.0), axis=0),
        out=np.full(n_voxels, np.nan),
        where=np.sum(np.where(finite_r2, r2_weights, 0.0), axis=0) > 0,
    ).astype(np.float32)
    return _ObservedFit(
        mean_correlation=mean_corr,
        mean_r2=mean_r2,
        outer_correlation=outer_corr,
        outer_r2=outer_r2,
        outer_alpha=outer_alpha,
        outer_pca=outer_pca,
        observed=tuple(observed),
        predicted=tuple(predictions),
        test_run_indices=tuple(test_run_indices),
        test_source_rows=tuple(test_source_rows),
    )


def _mode_smallest(values: np.ndarray) -> np.ndarray:
    output = np.full(values.shape[1], np.nan, dtype=np.float32)
    for voxel in range(values.shape[1]):
        finite = values[:, voxel][np.isfinite(values[:, voxel])]
        if finite.size:
            unique, counts = np.unique(finite, return_counts=True)
            output[voxel] = unique[np.argmax(counts)]
    return output


def fit_encoding(
    data: SubjectData,
    config: EncodingConfig,
    cv_plan: Optional[CVPlan] = None,
) -> EncodingResult:
    """Fit nested encoding CV and optional permutation nulls.

    Pass an explicit ``CVPlan`` for a project-specific split design. Otherwise
    the plan is built from ``config.cv_scheme``.
    """

    config.validate()
    cv_plan = cv_plan or build_cv_plan(data.runs, config)
    validate_cv_plan(cv_plan, data.runs)
    observed = _fit_observed(data.runs, config, cv_plan)
    observed_z = np.arctanh(
        np.clip(observed.mean_correlation.astype(np.float64), -0.999999, 0.999999)
    )
    n_voxels = observed.mean_correlation.size
    null_scores = np.empty((config.n_permutations, n_voxels), dtype=np.float32)
    rng = np.random.default_rng(config.random_seed)
    for permutation in range(config.n_permutations):
        permuted_runs = [
            EncodingRun(
                run_id=run.run_id,
                features=block_permute_rows(
                    run.features,
                    config.permutation_block_length,
                    rng,
                ),
                bold=run.bold,
                nuisance=run.nuisance,
                source_rows=run.source_rows,
            )
            for run in data.runs
        ]
        permuted = _fit_observed(permuted_runs, config, cv_plan)
        null_scores[permutation] = np.arctanh(
            np.clip(permuted.mean_correlation, -0.999999, 0.999999)
        )

    if config.n_permutations:
        null_mean = np.mean(null_scores, axis=0, dtype=np.float64)
        null_std = np.std(null_scores, axis=0, ddof=0, dtype=np.float64)
        effect = np.divide(
            observed_z - null_mean,
            null_std,
            out=np.full(n_voxels, np.nan),
            where=null_std > 0,
        )
        empirical_p = (
            np.sum(np.abs(null_scores) >= np.abs(observed_z)[None, :], axis=0) + 1.0
        ) / (config.n_permutations + 1.0)
        signed_z = np.sign(observed_z - null_mean) * norm.isf(empirical_p / 2.0)
    else:
        null_mean = np.full(n_voxels, np.nan)
        null_std = np.full(n_voxels, np.nan)
        effect = np.full(n_voxels, np.nan)
        empirical_p = np.full(n_voxels, np.nan)
        signed_z = np.full(n_voxels, np.nan)

    mean_alpha = np.nanmean(observed.outer_alpha, axis=0).astype(np.float32)
    return EncodingResult(
        subject_id=data.subject_id,
        run_ids=tuple(run.run_id for run in data.runs),
        fold_ids=tuple(fold.fold_id for fold in cv_plan.outer_folds),
        mean_correlation=observed.mean_correlation,
        mean_r2=observed.mean_r2,
        outer_fold_correlation=observed.outer_correlation,
        outer_fold_r2=observed.outer_r2,
        outer_selected_alpha=observed.outer_alpha,
        outer_selected_pca_components=observed.outer_pca,
        mean_selected_alpha=mean_alpha,
        mode_selected_pca_components=_mode_smallest(observed.outer_pca),
        observed_by_fold=observed.observed,
        predicted_by_fold=observed.predicted,
        test_run_indices_by_fold=observed.test_run_indices,
        test_source_rows_by_fold=observed.test_source_rows,
        null_mean_fisher_z=np.asarray(null_mean, dtype=np.float32),
        null_std_fisher_z=np.asarray(null_std, dtype=np.float32),
        null_standardized_effect_fisher_z=np.asarray(effect, dtype=np.float32),
        empirical_p=np.asarray(empirical_p, dtype=np.float32),
        signed_z=np.asarray(signed_z, dtype=np.float32),
        metadata={
            "schema_version": "encoding_v1",
            "cv_scheme": cv_plan.scheme,
            "outer_splits": len(cv_plan.outer_folds),
            "inner_splits": [len(fold.inner_folds) for fold in cv_plan.outer_folds],
            "cv_shuffle": config.cv_shuffle,
            "cv_seed": config.cv_seed,
            "embargo_rows": config.embargo_rows,
            "folds": [
                {
                    "fold_id": fold.fold_id,
                    "train": selection_metadata(fold.train, data.runs),
                    "test": selection_metadata(fold.test, data.runs),
                    "inner_folds": [
                        {
                            "fold_id": inner.fold_id,
                            "train": selection_metadata(inner.train, data.runs),
                            "validation": selection_metadata(
                                inner.validation, data.runs
                            ),
                        }
                        for inner in fold.inner_folds
                    ],
                }
                for fold in cv_plan.outer_folds
            ],
            "n_runs": len(data.runs),
            "run_ids": [run.run_id for run in data.runs],
            "run_row_counts": [int(run.bold.shape[0]) for run in data.runs],
            "lags": list(config.lags),
            "lag_mode": config.lag_mode,
            "ridge_alphas": list(config.ridge_alphas),
            "pca_components": [value if value is not None else 0 for value in config.pca_components],
            "pca_zero_means_disabled": True,
            "fisher_z_selection": config.fisher_z_selection,
            "n_permutations": config.n_permutations,
            "permutation_method": "within_run_block_shuffle_lagged_features",
            "permutation_block_length": config.permutation_block_length,
            "random_seed": config.random_seed,
            "nuisance_columns": list(config.nuisance_columns),
            "target_nuisance_scope": "complete_run_before_feature_alignment",
            "add_run_intercept": config.add_run_intercept,
            "mask_strategy": config.mask_strategy,
            "analysis_mask_path": config.analysis_mask_path,
        },
    )
