"""Nested temporal decoding for continuous scalar targets.

This module is deliberately separate from the legacy IID decoder. It keeps
story/run boundaries explicit, supports purged blocked folds, and accepts
precomputed target permutations so domain-specific code can permute a raw
stimulus variable before transformations such as HRF convolution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np
from sklearn.decomposition import PCA


Selection = Tuple[np.ndarray, ...]


@dataclass(frozen=True)
class TemporalRun:
    run_id: str
    features: np.ndarray
    source_rows: np.ndarray


@dataclass(frozen=True)
class TemporalInnerFold:
    fold_id: str
    train: Selection
    validation: Selection


@dataclass(frozen=True)
class TemporalOuterFold:
    fold_id: str
    train: Selection
    test: Selection
    inner_folds: Tuple[TemporalInnerFold, ...]


@dataclass(frozen=True)
class TemporalCVPlan:
    scheme: str
    outer_folds: Tuple[TemporalOuterFold, ...]


@dataclass(frozen=True)
class TemporalDecoderConfig:
    pca_components: Tuple[int, ...] = (2, 4, 8, 16, 32, 64, 128)
    ridge_alphas: Tuple[float, ...] = tuple(float(value) for value in np.logspace(-2, 6, 9))
    fisher_z_selection: bool = True
    random_seed: int = 0

    def validate(self) -> None:
        if not self.pca_components or any(value <= 0 for value in self.pca_components):
            raise ValueError("pca_components must contain positive integers")
        if not self.ridge_alphas or any(value < 0 for value in self.ridge_alphas):
            raise ValueError("ridge_alphas must contain non-negative values")


@dataclass
class TemporalDecodingResult:
    correlation: np.ndarray
    normalized_rmse: np.ndarray
    outer_fold_correlation: np.ndarray
    story_correlation: np.ndarray
    selected_pca_components: np.ndarray
    selected_alpha: np.ndarray
    observed_predictions: Tuple[np.ndarray, ...]
    empirical_p: float
    null_mean_fisher_z: float
    null_std_fisher_z: float
    null_standardized_effect_fisher_z: float
    metadata: dict


def _whole_runs(runs: Sequence[TemporalRun], selected: Sequence[int]) -> Selection:
    selected_set = set(selected)
    return tuple(
        np.arange(run.features.shape[0], dtype=np.int32)
        if index in selected_set
        else np.empty(0, dtype=np.int32)
        for index, run in enumerate(runs)
    )


def build_loso_plan(runs: Sequence[TemporalRun]) -> TemporalCVPlan:
    """Leave one complete story/run out, with nested leave-one-run-out CV."""

    if len(runs) < 4:
        raise ValueError("LOSO decoding requires at least four runs")
    outer_folds = []
    all_indices = tuple(range(len(runs)))
    for test_index in all_indices:
        train_indices = tuple(index for index in all_indices if index != test_index)
        inner = tuple(
            TemporalInnerFold(
                fold_id=f"validation-{runs[validation_index].run_id}",
                train=_whole_runs(
                    runs,
                    [index for index in train_indices if index != validation_index],
                ),
                validation=_whole_runs(runs, [validation_index]),
            )
            for validation_index in train_indices
        )
        outer_folds.append(
            TemporalOuterFold(
                fold_id=f"test-{runs[test_index].run_id}",
                train=_whole_runs(runs, train_indices),
                test=_whole_runs(runs, [test_index]),
                inner_folds=inner,
            )
        )
    plan = TemporalCVPlan("loso", tuple(outer_folds))
    validate_temporal_plan(plan, runs)
    return plan


def _purge(
    candidates: np.ndarray,
    heldout: np.ndarray,
    source_rows: np.ndarray,
    distance: int,
) -> np.ndarray:
    if candidates.size == 0 or heldout.size == 0 or distance == 0:
        return candidates
    heldout_source = source_rows[heldout]
    candidate_source = source_rows[candidates]
    low = int(heldout_source.min()) - distance
    high = int(heldout_source.max()) + distance
    return candidates[(candidate_source < low) | (candidate_source > high)]


def build_pooled_4x3_plan(
    runs: Sequence[TemporalRun], *, purge_rows: int = 24
) -> TemporalCVPlan:
    """Four synchronized story blocks with three nested validation blocks."""

    if len(runs) != 4:
        raise ValueError("pooled_4x3 decoding requires exactly four story runs")
    blocks = tuple(
        tuple(np.asarray(block, dtype=np.int32) for block in np.array_split(
            np.arange(run.features.shape[0], dtype=np.int32), 4
        ))
        for run in runs
    )
    outer_folds = []
    for test_block in range(4):
        test = tuple(run_blocks[test_block] for run_blocks in blocks)
        outer_train = tuple(
            _purge(
                np.concatenate([run_blocks[index] for index in range(4) if index != test_block]),
                test[run_index],
                runs[run_index].source_rows,
                purge_rows,
            ).astype(np.int32)
            for run_index, run_blocks in enumerate(blocks)
        )
        inner_folds = []
        for validation_block in (index for index in range(4) if index != test_block):
            validation = tuple(
                np.intersect1d(run_blocks[validation_block], outer_train[run_index]).astype(np.int32)
                for run_index, run_blocks in enumerate(blocks)
            )
            inner_train = tuple(
                _purge(
                    np.setdiff1d(outer_train[run_index], validation[run_index]).astype(np.int32),
                    validation[run_index],
                    runs[run_index].source_rows,
                    purge_rows,
                ).astype(np.int32)
                for run_index in range(len(runs))
            )
            inner_folds.append(
                TemporalInnerFold(
                    fold_id=f"inner-block-{validation_block + 1}",
                    train=inner_train,
                    validation=validation,
                )
            )
        outer_folds.append(
            TemporalOuterFold(
                fold_id=f"outer-block-{test_block + 1}",
                train=outer_train,
                test=test,
                inner_folds=tuple(inner_folds),
            )
        )
    plan = TemporalCVPlan("pooled_4x3", tuple(outer_folds))
    validate_temporal_plan(plan, runs)
    return plan


def validate_temporal_plan(plan: TemporalCVPlan, runs: Sequence[TemporalRun]) -> None:
    if not plan.outer_folds:
        raise ValueError("CV plan contains no outer folds")
    coverage = [list() for _ in runs]
    for outer in plan.outer_folds:
        if not outer.inner_folds:
            raise ValueError(f"{outer.fold_id} contains no inner folds")
        for run_index, run in enumerate(runs):
            train = np.asarray(outer.train[run_index])
            test = np.asarray(outer.test[run_index])
            if np.intersect1d(train, test).size:
                raise ValueError(f"{outer.fold_id} has overlapping train/test rows")
            if test.size:
                coverage[run_index].append(test)
            for rows in (train, test):
                if rows.size and (rows.min() < 0 or rows.max() >= run.features.shape[0]):
                    raise ValueError(f"{outer.fold_id} contains out-of-range rows")
        for inner in outer.inner_folds:
            for run_index in range(len(runs)):
                train = inner.train[run_index]
                validation = inner.validation[run_index]
                if np.intersect1d(train, validation).size:
                    raise ValueError(f"{inner.fold_id} has overlapping rows")
                if not np.all(np.isin(train, outer.train[run_index])) or not np.all(
                    np.isin(validation, outer.train[run_index])
                ):
                    raise ValueError(f"{inner.fold_id} is not contained in outer training rows")
    for run, heldout in zip(runs, coverage):
        if not heldout:
            raise ValueError(f"No held-out rows for {run.run_id}")
        combined = np.concatenate(heldout)
        if not np.array_equal(np.sort(combined), np.arange(run.features.shape[0])):
            raise ValueError(f"Held-out folds do not cover {run.run_id} exactly once")


def _stack(
    arrays: Sequence[np.ndarray], selection: Selection
) -> tuple[np.ndarray, np.ndarray]:
    selected = []
    run_labels = []
    for run_index, (values, rows) in enumerate(zip(arrays, selection)):
        if rows.size:
            selected.append(values[rows])
            run_labels.append(np.full(rows.size, run_index, dtype=np.int16))
    if not selected:
        raise ValueError("Cannot stack an empty temporal selection")
    return np.concatenate(selected, axis=0), np.concatenate(run_labels)


def _standardize_train_apply(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    safe = np.where(scale > 1e-7, scale, 1.0)
    train_z = (train - mean) / safe
    test_z = (test - mean) / safe
    train_z[:, scale <= 1e-7] = 0.0
    test_z[:, scale <= 1e-7] = 0.0
    return train_z.astype(np.float32), test_z.astype(np.float32)


def _column_correlation(observed: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    left = observed - observed.mean(axis=0, keepdims=True)
    right = predicted - predicted.mean(axis=0, keepdims=True)
    numerator = np.sum(left * right, axis=0)
    denominator = np.sqrt(np.sum(left * left, axis=0) * np.sum(right * right, axis=0))
    return np.divide(
        numerator,
        denominator,
        out=np.full(numerator.shape, np.nan, dtype=np.float64),
        where=denominator > 0,
    )


def _score_cells(
    observed: np.ndarray, predicted: np.ndarray, run_labels: np.ndarray
) -> tuple[np.ndarray, float]:
    correlations = []
    weights = []
    for run_index in np.unique(run_labels):
        selected = run_labels == run_index
        correlations.append(_column_correlation(observed[selected], predicted[selected]))
        weights.append(max(int(selected.sum()) - 3, 1))
    values = np.stack(correlations)
    weights_array = np.asarray(weights, dtype=float)[:, None]
    finite = np.isfinite(values)
    z_values = np.arctanh(np.clip(values, -0.999999, 0.999999))
    numerator = np.sum(np.where(finite, z_values * weights_array, 0.0), axis=0)
    denominator = np.sum(np.where(finite, weights_array, 0.0), axis=0)
    mean_z = np.divide(
        numerator,
        denominator,
        out=np.full(numerator.shape, np.nan),
        where=denominator > 0,
    )
    return np.tanh(mean_z), float(np.sum(weights))


def _prepare_pca(
    train: np.ndarray,
    test: np.ndarray,
    maximum_components: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    train_z, test_z = _standardize_train_apply(train, test)
    maximum = min(maximum_components, train_z.shape[0] - 1, train_z.shape[1])
    if maximum < 1:
        raise ValueError("Not enough rows/features to fit PCA")
    solver = "full" if maximum == min(train_z.shape) else "randomized"
    model = PCA(n_components=maximum, svd_solver=solver, random_state=seed)
    return (
        model.fit_transform(train_z).astype(np.float32),
        model.transform(test_z).astype(np.float32),
        maximum,
    )


def _ridge_predict_many(
    train: np.ndarray,
    targets: np.ndarray,
    test: np.ndarray,
    alpha: float,
) -> np.ndarray:
    gram = train.T @ train
    gram.flat[:: gram.shape[0] + 1] += float(alpha)
    coefficients = np.linalg.solve(gram, train.T @ targets)
    return test @ coefficients


def _fit_fold(
    feature_runs: Sequence[np.ndarray],
    target_runs: Sequence[np.ndarray],
    train_selection: Selection,
    test_selection: Selection,
    components: int,
    alpha: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    x_train, _ = _stack(feature_runs, train_selection)
    x_test, test_labels = _stack(feature_runs, test_selection)
    y_train, _ = _stack(target_runs, train_selection)
    y_test, _ = _stack(target_runs, test_selection)
    x_fit, x_predict, maximum = _prepare_pca(x_train, x_test, components, seed)
    if components > maximum:
        raise ValueError("Selected PCA component count exceeds fold rank")
    y_mean = y_train.mean(axis=0)
    y_scale = y_train.std(axis=0)
    y_scale = np.where(y_scale > 1e-7, y_scale, 1.0)
    y_train_z = (y_train - y_mean) / y_scale
    prediction_z = _ridge_predict_many(
        x_fit[:, :components], y_train_z, x_predict[:, :components], alpha
    )
    prediction = prediction_z * y_scale + y_mean
    score, weight = _score_cells(y_test, prediction, test_labels)
    return score, y_test, prediction, test_labels, weight


def fit_temporal_decoder(
    runs: Sequence[TemporalRun],
    target_variants: Sequence[np.ndarray],
    plan: TemporalCVPlan,
    config: TemporalDecoderConfig = TemporalDecoderConfig(),
    *,
    evaluation_runs: Sequence[TemporalRun] | None = None,
) -> TemporalDecodingResult:
    """Fit observed and permuted scalar targets through identical nested CV.

    Each target array must have shape ``(run_rows, n_variants)``. Column zero
    is observed and remaining columns are null targets.
    """

    config.validate()
    validate_temporal_plan(plan, runs)
    feature_runs = [np.asarray(run.features, dtype=np.float32) for run in runs]
    evaluation_runs = runs if evaluation_runs is None else evaluation_runs
    if len(evaluation_runs) != len(runs):
        raise ValueError("evaluation_runs must match the training run count")
    for source, evaluation in zip(runs, evaluation_runs):
        if (
            source.run_id != evaluation.run_id
            or source.features.shape != evaluation.features.shape
            or not np.array_equal(source.source_rows, evaluation.source_rows)
        ):
            raise ValueError("evaluation_runs must have matching run IDs, rows, and feature shapes")
    evaluation_feature_runs = [
        np.asarray(run.features, dtype=np.float32) for run in evaluation_runs
    ]
    targets = [np.asarray(values, dtype=np.float64) for values in target_variants]
    n_variants = targets[0].shape[1]
    if any(values.shape != (run.features.shape[0], n_variants) for run, values in zip(runs, targets)):
        raise ValueError("Target variants must match run rows and share a column count")
    candidates = tuple(
        (components, alpha)
        for components in config.pca_components
        for alpha in config.ridge_alphas
    )
    n_outer = len(plan.outer_folds)
    outer_scores = np.full((n_outer, n_variants), np.nan)
    outer_nrmse = np.full_like(outer_scores, np.nan)
    selected_components = np.full((n_outer, n_variants), np.nan)
    selected_alpha = np.full_like(selected_components, np.nan)
    outer_weights = np.zeros(n_outer, dtype=float)
    observed_predictions = [np.full(run.features.shape[0], np.nan) for run in runs]

    for outer_index, outer in enumerate(plan.outer_folds):
        inner_scores = np.full((len(outer.inner_folds), len(candidates), n_variants), np.nan)
        inner_weights = np.zeros(len(outer.inner_folds), dtype=float)
        for inner_index, inner in enumerate(outer.inner_folds):
            x_train, _ = _stack(feature_runs, inner.train)
            x_validation, validation_labels = _stack(feature_runs, inner.validation)
            y_train, _ = _stack(targets, inner.train)
            y_validation, _ = _stack(targets, inner.validation)
            x_fit, x_predict, maximum = _prepare_pca(
                x_train,
                x_validation,
                max(config.pca_components),
                config.random_seed + outer_index * 10 + inner_index,
            )
            y_mean = y_train.mean(axis=0)
            y_scale = np.where(y_train.std(axis=0) > 1e-7, y_train.std(axis=0), 1.0)
            y_train_z = (y_train - y_mean) / y_scale
            for candidate_index, (components, alpha) in enumerate(candidates):
                if components > maximum:
                    continue
                prediction = _ridge_predict_many(
                    x_fit[:, :components], y_train_z, x_predict[:, :components], alpha
                )
                prediction = prediction * y_scale + y_mean
                score, weight = _score_cells(y_validation, prediction, validation_labels)
                inner_scores[inner_index, candidate_index] = score
                inner_weights[inner_index] = weight

        selection_values = np.arctanh(np.clip(inner_scores, -0.999999, 0.999999))
        if not config.fisher_z_selection:
            selection_values = inner_scores
        weights = inner_weights[:, None, None]
        finite = np.isfinite(selection_values)
        numerator = np.sum(np.where(finite, selection_values * weights, 0.0), axis=0)
        denominator = np.sum(np.where(finite, weights, 0.0), axis=0)
        averaged = np.divide(
            numerator,
            denominator,
            out=np.full(numerator.shape, -np.inf),
            where=denominator > 0,
        )
        winners = np.argmax(averaged, axis=0)

        x_train, _ = _stack(feature_runs, outer.train)
        x_test, test_labels = _stack(evaluation_feature_runs, outer.test)
        y_train, _ = _stack(targets, outer.train)
        y_test, _ = _stack(targets, outer.test)
        x_fit, x_predict, maximum = _prepare_pca(
            x_train, x_test, max(config.pca_components), config.random_seed + 100 + outer_index
        )
        y_mean = y_train.mean(axis=0)
        y_scale = np.where(y_train.std(axis=0) > 1e-7, y_train.std(axis=0), 1.0)
        y_train_z = (y_train - y_mean) / y_scale
        prediction = np.full_like(y_test, np.nan)
        for candidate_index, (components, alpha) in enumerate(candidates):
            selected = winners == candidate_index
            if not selected.any():
                continue
            if components > maximum:
                raise ValueError("Inner CV selected a PCA size unavailable to the outer fold")
            predicted_z = _ridge_predict_many(
                x_fit[:, :components],
                y_train_z[:, selected],
                x_predict[:, :components],
                alpha,
            )
            prediction[:, selected] = predicted_z * y_scale[selected] + y_mean[selected]
            selected_components[outer_index, selected] = components
            selected_alpha[outer_index, selected] = alpha
        outer_scores[outer_index], outer_weights[outer_index] = _score_cells(
            y_test, prediction, test_labels
        )
        rmse = np.sqrt(np.mean((y_test - prediction) ** 2, axis=0))
        observed_scale = np.std(y_test, axis=0)
        outer_nrmse[outer_index] = np.divide(
            rmse,
            observed_scale,
            out=np.full(rmse.shape, np.nan),
            where=observed_scale > 1e-7,
        )
        offset = 0
        for run_index, rows in enumerate(outer.test):
            if rows.size:
                observed_predictions[run_index][rows] = prediction[offset : offset + rows.size, 0]
                offset += rows.size

    z_scores = np.arctanh(np.clip(outer_scores, -0.999999, 0.999999))
    aggregate_z = np.average(z_scores, axis=0, weights=outer_weights)
    aggregate_correlation = np.tanh(aggregate_z)
    aggregate_nrmse = np.average(outer_nrmse, axis=0, weights=outer_weights)

    story_scores = np.full((len(runs), n_variants), np.nan)
    for run_index, run_targets in enumerate(targets):
        predictions = np.full_like(run_targets, np.nan)
        for outer_index, outer in enumerate(plan.outer_folds):
            rows = outer.test[run_index]
            if not rows.size:
                continue
            # Refit only for story diagnostics would duplicate all production work.
            # The observed trace is retained exactly; null story scores are not needed.
            if n_variants and np.isfinite(observed_predictions[run_index][rows]).all():
                predictions[rows, 0] = observed_predictions[run_index][rows]
        story_scores[run_index, 0] = _column_correlation(
            run_targets[:, :1], predictions[:, :1]
        )[0]

    null_z = aggregate_z[1:]
    observed_z = float(aggregate_z[0])
    if null_z.size:
        null_mean = float(np.mean(null_z))
        null_std = float(np.std(null_z))
        empirical_p = float((np.sum(null_z >= observed_z) + 1) / (null_z.size + 1))
        standardized = (observed_z - null_mean) / null_std if null_std > 0 else np.nan
    else:
        null_mean = null_std = empirical_p = standardized = np.nan
    return TemporalDecodingResult(
        correlation=aggregate_correlation.astype(np.float32),
        normalized_rmse=aggregate_nrmse.astype(np.float32),
        outer_fold_correlation=outer_scores.astype(np.float32),
        story_correlation=story_scores.astype(np.float32),
        selected_pca_components=selected_components.astype(np.float32),
        selected_alpha=selected_alpha.astype(np.float32),
        observed_predictions=tuple(values.astype(np.float32) for values in observed_predictions),
        empirical_p=empirical_p,
        null_mean_fisher_z=null_mean,
        null_std_fisher_z=null_std,
        null_standardized_effect_fisher_z=float(standardized),
        metadata={
            "schema_version": "temporal_continuous_decoding_v1",
            "cv_scheme": plan.scheme,
            "fold_ids": [fold.fold_id for fold in plan.outer_folds],
            "run_ids": [run.run_id for run in runs],
            "pca_components": list(config.pca_components),
            "ridge_alphas": list(config.ridge_alphas),
            "n_target_variants": n_variants,
            "null_iterations": max(n_variants - 1, 0),
            "selection_metric": "weighted_fisher_z_pearson_correlation",
            "evaluation_features": "training_runs" if evaluation_runs is runs else "separate_evaluation_runs",
        },
    )
