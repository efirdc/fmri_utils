from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
from sklearn.model_selection import KFold

from .config import EncodingConfig
from .data import EncodingRun


Selection = Tuple[np.ndarray, ...]


@dataclass(frozen=True)
class InnerFold:
    fold_id: str
    train: Selection
    validation: Selection


@dataclass(frozen=True)
class OuterFold:
    fold_id: str
    train: Selection
    test: Selection
    inner_folds: Tuple[InnerFold, ...]


@dataclass(frozen=True)
class CVPlan:
    scheme: str
    outer_folds: Tuple[OuterFold, ...]


def _empty_selection(n_runs: int) -> List[np.ndarray]:
    return [np.empty(0, dtype=np.int32) for _ in range(n_runs)]


def _whole_runs(runs: Sequence[EncodingRun], selected: Sequence[int]) -> Selection:
    wanted = set(int(index) for index in selected)
    return tuple(
        np.arange(run.bold.shape[0], dtype=np.int32)
        if run_index in wanted
        else np.empty(0, dtype=np.int32)
        for run_index, run in enumerate(runs)
    )


def _validate_selection(selection: Selection, label: str) -> None:
    if sum(rows.size for rows in selection) == 0:
        raise ValueError("{} has no rows".format(label))


def _leave_one_run_out(runs: Sequence[EncodingRun]) -> CVPlan:
    if len(runs) < 3:
        raise ValueError("leave_one_run_out requires at least three runs")
    outer_folds = []
    all_indices = list(range(len(runs)))
    for test_index in all_indices:
        train_indices = [index for index in all_indices if index != test_index]
        inner = []
        for validation_index in train_indices:
            inner_train = [index for index in train_indices if index != validation_index]
            inner.append(
                InnerFold(
                    fold_id="validation-{}".format(runs[validation_index].run_id),
                    train=_whole_runs(runs, inner_train),
                    validation=_whole_runs(runs, [validation_index]),
                )
            )
        outer_folds.append(
            OuterFold(
                fold_id="test-{}".format(runs[test_index].run_id),
                train=_whole_runs(runs, train_indices),
                test=_whole_runs(runs, [test_index]),
                inner_folds=tuple(inner),
            )
        )
    return CVPlan("leave_one_run_out", tuple(outer_folds))


def _kfold_indices(
    values: np.ndarray,
    n_splits: int,
    *,
    shuffle: bool,
    seed: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    if n_splits > values.size:
        raise ValueError(
            "Requested {} folds for only {} run groups".format(n_splits, values.size)
        )
    splitter = KFold(
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=seed if shuffle else None,
    )
    return [(values[train], values[test]) for train, test in splitter.split(values)]


def _grouped_run_kfold(
    runs: Sequence[EncodingRun], config: EncodingConfig
) -> CVPlan:
    run_indices = np.arange(len(runs), dtype=np.int32)
    outer_folds = []
    for outer_index, (train_runs, test_runs) in enumerate(
        _kfold_indices(
            run_indices,
            config.outer_splits,
            shuffle=config.cv_shuffle,
            seed=config.cv_seed,
        )
    ):
        inner = []
        for inner_index, (inner_train, validation) in enumerate(
            _kfold_indices(
                train_runs,
                config.inner_splits,
                shuffle=config.cv_shuffle,
                seed=config.cv_seed + outer_index + 1,
            )
        ):
            inner.append(
                InnerFold(
                    fold_id="inner-{:02d}".format(inner_index + 1),
                    train=_whole_runs(runs, inner_train),
                    validation=_whole_runs(runs, validation),
                )
            )
        outer_folds.append(
            OuterFold(
                fold_id="outer-{:02d}".format(outer_index + 1),
                train=_whole_runs(runs, train_runs),
                test=_whole_runs(runs, test_runs),
                inner_folds=tuple(inner),
            )
        )
    return CVPlan("grouped_run_kfold", tuple(outer_folds))


def _purge_near_heldout(
    candidates: np.ndarray,
    heldout: np.ndarray,
    source_rows: np.ndarray,
    embargo: int,
) -> np.ndarray:
    if embargo == 0 or candidates.size == 0 or heldout.size == 0:
        return candidates
    candidate_rows = source_rows[candidates]
    heldout_rows = source_rows[heldout]
    lower = int(np.min(heldout_rows)) - embargo
    upper = int(np.max(heldout_rows)) + embargo
    return candidates[(candidate_rows < lower) | (candidate_rows > upper)]


def _blocked_kfold(runs: Sequence[EncodingRun], config: EncodingConfig) -> CVPlan:
    if config.inner_splits > config.outer_splits - 1:
        raise ValueError("blocked_kfold requires inner_splits <= outer_splits - 1")
    blocks = []
    for run in runs:
        if run.bold.shape[0] < config.outer_splits:
            raise ValueError(
                "Run {} has fewer rows than outer_splits".format(run.run_id)
            )
        blocks.append(
            tuple(
                np.asarray(rows, dtype=np.int32)
                for rows in np.array_split(
                    np.arange(run.bold.shape[0], dtype=np.int32), config.outer_splits
                )
            )
        )

    outer_folds = []
    all_block_ids = np.arange(config.outer_splits, dtype=np.int32)
    for outer_index in range(config.outer_splits):
        test = tuple(run_blocks[outer_index] for run_blocks in blocks)
        outer_train = []
        for run_index, run in enumerate(runs):
            candidates = np.concatenate(
                [blocks[run_index][i] for i in all_block_ids if i != outer_index]
            )
            outer_train.append(
                _purge_near_heldout(
                    candidates,
                    test[run_index],
                    run.source_rows,
                    config.embargo_rows,
                )
            )
        outer_train_selection = tuple(outer_train)
        remaining_ids = all_block_ids[all_block_ids != outer_index]
        validation_groups = np.array_split(remaining_ids, config.inner_splits)
        inner_folds = []
        for inner_index, validation_ids in enumerate(validation_groups):
            validation = []
            inner_train = []
            for run_index, run in enumerate(runs):
                validation_rows = np.concatenate(
                    [blocks[run_index][i] for i in validation_ids]
                )
                validation_rows = np.intersect1d(
                    validation_rows, outer_train_selection[run_index], assume_unique=True
                ).astype(np.int32)
                candidates = np.setdiff1d(
                    outer_train_selection[run_index], validation_rows, assume_unique=True
                ).astype(np.int32)
                inner_train_rows = _purge_near_heldout(
                    candidates,
                    validation_rows,
                    run.source_rows,
                    config.embargo_rows,
                )
                validation.append(validation_rows)
                inner_train.append(inner_train_rows)
            inner = InnerFold(
                fold_id="inner-{:02d}".format(inner_index + 1),
                train=tuple(inner_train),
                validation=tuple(validation),
            )
            _validate_selection(inner.train, inner.fold_id + " training")
            _validate_selection(inner.validation, inner.fold_id + " validation")
            inner_folds.append(inner)
        fold = OuterFold(
            fold_id="outer-{:02d}".format(outer_index + 1),
            train=outer_train_selection,
            test=test,
            inner_folds=tuple(inner_folds),
        )
        _validate_selection(fold.train, fold.fold_id + " training")
        _validate_selection(fold.test, fold.fold_id + " test")
        outer_folds.append(fold)
    return CVPlan("blocked_kfold", tuple(outer_folds))


def build_cv_plan(runs: Sequence[EncodingRun], config: EncodingConfig) -> CVPlan:
    """Build deterministic nested train/validation/test row selections."""

    if config.cv_scheme == "leave_one_run_out":
        plan = _leave_one_run_out(runs)
    elif config.cv_scheme == "grouped_run_kfold":
        plan = _grouped_run_kfold(runs, config)
    else:
        plan = _blocked_kfold(runs, config)
    validate_cv_plan(plan, runs)
    return plan


def _validate_rows(
    selection: Selection,
    runs: Sequence[EncodingRun],
    label: str,
) -> None:
    if len(selection) != len(runs):
        raise ValueError("{} must contain one row array per run".format(label))
    for run, rows in zip(runs, selection):
        values = np.asarray(rows)
        if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
            raise ValueError("{} rows must be one-dimensional integer arrays".format(label))
        if np.unique(values).size != values.size:
            raise ValueError("{} contains duplicate rows for {}".format(label, run.run_id))
        if values.size and (values.min() < 0 or values.max() >= run.bold.shape[0]):
            raise ValueError("{} contains out-of-range rows for {}".format(label, run.run_id))


def validate_cv_plan(plan: CVPlan, runs: Sequence[EncodingRun]) -> None:
    """Validate a built-in or caller-supplied nested CV plan."""

    if not plan.outer_folds:
        raise ValueError("cv_plan must contain at least one outer fold")
    for outer in plan.outer_folds:
        _validate_rows(outer.train, runs, outer.fold_id + " training")
        _validate_rows(outer.test, runs, outer.fold_id + " test")
        _validate_selection(outer.train, outer.fold_id + " training")
        _validate_selection(outer.test, outer.fold_id + " test")
        if not outer.inner_folds:
            raise ValueError("{} must contain inner folds".format(outer.fold_id))
        for run_index in range(len(runs)):
            if np.intersect1d(outer.train[run_index], outer.test[run_index]).size:
                raise ValueError("{} has overlapping train/test rows".format(outer.fold_id))
        for inner in outer.inner_folds:
            _validate_rows(inner.train, runs, inner.fold_id + " training")
            _validate_rows(inner.validation, runs, inner.fold_id + " validation")
            _validate_selection(inner.train, inner.fold_id + " training")
            _validate_selection(inner.validation, inner.fold_id + " validation")
            for run_index in range(len(runs)):
                if np.intersect1d(
                    inner.train[run_index], inner.validation[run_index]
                ).size:
                    raise ValueError("{} has overlapping train/validation rows".format(inner.fold_id))
                outer_rows = outer.train[run_index]
                if not np.all(np.isin(inner.train[run_index], outer_rows)) or not np.all(
                    np.isin(inner.validation[run_index], outer_rows)
                ):
                    raise ValueError(
                        "{} rows must be subsets of outer training rows".format(
                            inner.fold_id
                        )
                    )


def selection_metadata(selection: Selection, runs: Sequence[EncodingRun]) -> List[dict]:
    return [
        {
            "run_id": run.run_id,
            "n_rows": int(rows.size),
            "source_row_min": int(run.source_rows[rows].min()) if rows.size else None,
            "source_row_max": int(run.source_rows[rows].max()) if rows.size else None,
        }
        for run, rows in zip(runs, selection)
    ]
