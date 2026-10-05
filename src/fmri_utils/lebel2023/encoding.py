"""The encoding model: stimulus features -> each voxel's response, scored on the held-out story.

Design (``prepare_design``), all of it fitted on the training stories only:

1. z-score the features, then PCA to 256 components (randomized, seed 2023);
2. append any extra columns (the word rate) after the PCA, z-scored, so one regressor is not one
   of hundreds of PCA inputs smeared across components;
3. FIR delays of 1-4 TRs per story (zero-padded, nothing crosses a story boundary);
4. z-score the delayed design.

Fit (``fit_voxelwise``): ridge regression per voxel through ``fmri_utils.encoding.fit_encoding``,
alpha chosen per voxel from 10 ... 1e5 (half-decade steps) by 5-fold cross-validation over whole
training stories (story i in fold i mod 5), scored by Fisher-z-averaged validation r; then refit on
every training story and scored by Pearson r on the test story ``wheretheressmoke`` (the released
mean of its repeats). The noise ceiling comes from the test story's repeats; normalised r is
r / max(ceiling, 0.3).

A subject is fitted in voxel chunks (contiguous column slabs; every voxelwise quantity depends only
on that voxel, so chunks stitch into exactly the whole-brain fit). Each chunk writes
``<output>/<model>/<subject>/chunks/chunk_<c>_of_<n>.npz`` with ``voxel_index``, ``correlation_raw``,
``correlation_noise_ceiling_normalized``, ``noise_ceiling`` and ``selected_alpha``; ``stitch``
gathers them into column vectors and native NIfTI maps.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .dataset import FIR_DELAYS, RIDGE_ALPHAS, TEST_STORY, Dataset, common_finite
from . import features as feature_io

NOISE_CEILING_FLOOR = 0.3
PCA_COMPONENTS = 256
METRICS = ("correlation_raw", "correlation_noise_ceiling_normalized", "noise_ceiling", "selected_alpha")
CHUNK = re.compile(r"chunk_(\d{3})_of_(\d{3})\.npz$")


@dataclass(frozen=True)
class Design:
    train: np.ndarray          # training rows (all training stories stacked) x design columns
    test: np.ndarray           # test-story rows x design columns
    story_groups: np.ndarray   # training story index per training row
    pca_components: int


def delayed(matrix: np.ndarray, delays=FIR_DELAYS) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    blocks = []
    for delay in delays:
        block = np.zeros_like(matrix)
        block[delay:] = matrix[:-delay]
        blocks.append(block)
    return np.hstack(blocks)


def _zscore_with(train: np.ndarray):
    mean = train.mean(axis=0, dtype=np.float64)
    std = train.std(axis=0, dtype=np.float64)
    std[std < 1e-8] = 1.0
    return mean, std


def prepare_design(train_features: list[np.ndarray], test_features: np.ndarray, *, pca_components: int = PCA_COMPONENTS,
                   extra_train: list[np.ndarray] | None = None, extra_test: np.ndarray | None = None) -> Design:
    """z-score, PCA, append extras, FIR-delay, z-score (all fitted on the training stories)."""
    from sklearn.decomposition import PCA
    raw = np.vstack([np.asarray(m, dtype=np.float32) for m in train_features])
    mean, std = _zscore_with(raw)
    stacked = ((raw - mean) / std).astype(np.float32)
    test = ((np.asarray(test_features, dtype=np.float32) - mean) / std).astype(np.float32)
    n_components = min(int(pca_components), stacked.shape[0] - 1, stacked.shape[1])
    if n_components < 2:
        raise ValueError(f"design too small for PCA: {stacked.shape}")
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=2023).fit(stacked)
    reduced, start = [], 0
    for matrix in train_features:
        stop = start + matrix.shape[0]
        reduced.append(pca.transform(stacked[start:stop]).astype(np.float32))
        start = stop
    test_reduced = pca.transform(test).astype(np.float32)
    if extra_train is not None:
        if extra_test is None or len(extra_train) != len(reduced):
            raise ValueError("extra_train needs one matrix per training story, and extra_test")
        extra_mean, extra_std = _zscore_with(np.vstack([np.asarray(m, dtype=np.float32) for m in extra_train]))

        def append(block, extra):
            extra = np.asarray(extra, dtype=np.float32).reshape(block.shape[0], -1)
            return np.hstack([block, ((extra - extra_mean) / extra_std).astype(np.float32)])
        reduced = [append(r, e) for r, e in zip(reduced, extra_train)]
        test_reduced = append(test_reduced, extra_test)
    train_delayed = [delayed(m) for m in reduced]
    train = np.vstack(train_delayed)
    groups = np.concatenate([np.full(m.shape[0], i, dtype=np.int16) for i, m in enumerate(train_delayed)])
    mean, std = _zscore_with(train)
    return Design(train=((train - mean) / std).astype(np.float32),
                  test=((delayed(test_reduced) - mean) / std).astype(np.float32),
                  story_groups=groups, pca_components=n_components)


def design_from_blocks(train_blocks: list[np.ndarray], test_block: np.ndarray) -> Design:
    """A design used as is (lag 0), z-scored with the training statistics: cross-participant predictors."""
    raw = np.vstack(train_blocks)
    mean, sd = raw.mean(axis=0), raw.std(axis=0)
    sd[sd < 1e-8] = 1.0
    groups = np.concatenate([np.full(b.shape[0], i, dtype=np.int16) for i, b in enumerate(train_blocks)])
    return Design(train=((raw - mean) / sd).astype(np.float32), test=((test_block - mean) / sd).astype(np.float32),
                  story_groups=groups, pca_components=raw.shape[1])


def fixed_story_cv_plan(runs, inner_splits: int = 5):
    """The last run is the fixed test; whole training stories (story i in fold i mod k) tune alpha."""
    from fmri_utils.encoding import CVPlan, InnerFold, OuterFold
    if len(runs) < 4:
        raise ValueError("needs at least three training stories")
    n_train = len(runs) - 1
    k = min(int(inner_splits), n_train)

    def whole(indices: set[int]):
        return tuple(np.arange(run.bold.shape[0], dtype=np.int32) if i in indices else np.empty(0, dtype=np.int32)
                     for i, run in enumerate(runs))
    training = set(range(n_train))
    inner = tuple(InnerFold(fold_id=f"story-validation-{f + 1:02d}",
                            train=whole(training - {i for i in range(n_train) if i % k == f}),
                            validation=whole({i for i in range(n_train) if i % k == f}))
                  for f in range(k))
    outer = OuterFold(fold_id=f"test-{runs[-1].run_id}", train=whole(training), test=whole({n_train}), inner_folds=inner)
    return CVPlan("fixed_heldout_story", (outer,))


def fit_voxelwise(design: Design, stories: list[str], train_responses: list[np.ndarray], test_response: np.ndarray,
                  subject: str = "subject", alphas=RIDGE_ALPHAS) -> tuple[np.ndarray, np.ndarray]:
    """Per-voxel ridge with story-blocked CV for alpha; returns (test r, selected alpha) per column."""
    from fmri_utils.encoding import EncodingConfig, EncodingRun, SubjectData, fit_encoding
    runs, start = [], 0
    for story, response in zip(stories, train_responses):
        stop = start + response.shape[0]
        runs.append(EncodingRun(run_id=story, features=design.train[start:stop], bold=response,
                                nuisance=np.empty((response.shape[0], 0), dtype=np.float32),
                                source_rows=np.arange(response.shape[0], dtype=np.int32)))
        start = stop
    runs.append(EncodingRun(run_id=TEST_STORY, features=design.test, bold=test_response,
                            nuisance=np.empty((test_response.shape[0], 0), dtype=np.float32),
                            source_rows=np.arange(test_response.shape[0], dtype=np.int32)))
    data = SubjectData(subject_id=subject, runs=runs, mask=np.ones((test_response.shape[1], 1, 1), dtype=bool),
                       reference_image=None, voxel_indices=np.arange(test_response.shape[1]))
    config = EncodingConfig(lags=(0,), ridge_alphas=tuple(float(a) for a in alphas), pca_components=(None,),
                            add_run_intercept=False, n_permutations=0)
    result = fit_encoding(data, config, fixed_story_cv_plan(runs))
    return result.mean_correlation.astype(np.float32), result.mean_selected_alpha.astype(np.float32)


def noise_ceiling(repeats: np.ndarray) -> np.ndarray:
    """The correlation ceiling of the repeat mean (repeats x TRs x voxels), Schoppe et al.'s estimator."""
    values = np.asarray(repeats, dtype=np.float64)
    if values.ndim != 3 or values.shape[0] < 2:
        raise ValueError("repeats must be (n_repeats, n_times, n_voxels)")
    n = values.shape[0]
    total = np.var(values.mean(axis=0), axis=0, ddof=1)
    within = np.mean(np.var(values, axis=1, ddof=1), axis=0)
    signal = np.maximum((n * total - within) / (n - 1), 0.0)
    ratio = np.divide(signal, total, out=np.zeros_like(signal), where=total > 0)
    return np.sqrt(ratio).clip(0.0, 1.0).astype(np.float32)


def normalise(correlation: np.ndarray, ceiling: np.ndarray, floor: float = NOISE_CEILING_FLOOR) -> np.ndarray:
    return (np.asarray(correlation, dtype=np.float32) / np.maximum(np.asarray(ceiling, dtype=np.float32), floor)).astype(np.float32)


def chunk_bounds(n_columns: int, chunk: int, n_chunks: int) -> tuple[int, int]:
    edges = np.linspace(0, n_columns, n_chunks + 1).round().astype(int)
    if not 0 <= chunk < n_chunks:
        raise ValueError(f"chunk {chunk} outside 0..{n_chunks - 1}")
    return int(edges[chunk]), int(edges[chunk + 1])


def save_chunk(folder: Path, chunk: int, n_chunks: int, voxel_index: np.ndarray, arrays: dict, meta: dict) -> Path:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    stem = f"chunk_{chunk:03d}_of_{n_chunks:03d}"
    np.savez_compressed(folder / f"{stem}.npz", voxel_index=np.asarray(voxel_index, dtype=np.int64), **arrays)
    (folder / f"{stem}.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return folder / f"{stem}.npz"


def fit_features(dataset: Dataset, subject: str, feature_root: Path, feature: str, output: Path, *,
                 model: str | None = None, extra_root: Path | None = None, chunk: int = 0, n_chunks: int = 1,
                 pca_components: int = PCA_COMPONENTS) -> Path:
    """Fit one subject's voxel chunk on one feature space (optionally with an extra column).

    ``feature`` names ``<feature_root>/<feature>/<story>.npy``; ``extra_root`` is a folder of
    ``<story>.npy`` whose first column is appended after the PCA (the word rate). Writes the chunk to
    ``<output>/<model or feature>/<subject>/chunks/``."""
    stories = dataset.training_stories(subject)
    start, stop = chunk_bounds(dataset.n_columns(subject), chunk, n_chunks)
    train_features = [feature_io.load(feature_root, feature, s) for s in stories]
    test_features = feature_io.load(feature_root, feature, TEST_STORY)
    train = [dataset.response(subject, s, start, stop) for s in stories]
    test = dataset.response(subject, TEST_STORY, start, stop)
    repeats = dataset.repeats(subject, TEST_STORY, start, stop)
    for story, f, r in zip(stories, train_features, train):
        if f.shape[0] != r.shape[0]:
            raise ValueError(f"{subject} {story}: features {f.shape} vs response {r.shape}")
    if not np.allclose(repeats.mean(axis=0), test, rtol=1e-5, atol=1e-5, equal_nan=True):
        raise ValueError(f"{subject}: the released test response is not the repeat mean")
    valid = common_finite([*train, test, repeats])
    train, test, repeats = [r[:, valid] for r in train], test[:, valid], repeats[:, :, valid]
    extra_train = extra_test = None
    if extra_root is not None:
        def extra(story, rows):
            values = np.load(Path(extra_root) / f"{story}.npy").astype(np.float32)
            values = values.reshape(values.shape[0], -1)[:, :1]
            if values.shape[0] != rows:
                raise ValueError(f"{story}: extra column has {values.shape[0]} rows, not {rows}")
            return values
        extra_train = [extra(s, f.shape[0]) for s, f in zip(stories, train_features)]
        extra_test = extra(TEST_STORY, test_features.shape[0])
    design = prepare_design(train_features, test_features, pca_components=pca_components,
                            extra_train=extra_train, extra_test=extra_test)
    correlation, alpha = fit_voxelwise(design, stories, train, test, subject)
    ceiling = noise_ceiling(repeats)
    model = model or feature
    arrays = {"correlation_raw": correlation, "correlation_noise_ceiling_normalized": normalise(correlation, ceiling),
              "noise_ceiling": ceiling, "selected_alpha": alpha}
    meta = {"subject": subject, "model": model, "feature": feature, "extra_root": str(extra_root) if extra_root else None,
            "voxel_start": start, "voxel_stop": stop, "n_columns": dataset.n_columns(subject),
            "n_modeled_voxels": int(valid.sum()), "training_stories": stories, "test_story": TEST_STORY,
            "pca_components": design.pca_components, "fir_delays": list(FIR_DELAYS),
            "mean_r": float(np.nanmean(correlation))}
    path = save_chunk(Path(output) / model / subject / "chunks", chunk, n_chunks, start + np.flatnonzero(valid), arrays, meta)
    print(f"[{subject} {model} chunk {chunk}/{n_chunks}] mean r {np.nanmean(correlation):.4f}", flush=True)
    return path


def read_chunks(folder: Path, n_columns: int | None = None, metrics=METRICS) -> dict[str, np.ndarray]:
    """A model's chunks (``<model>/<subject>``) as column vectors (NaN where unmodelled).

    Raises if the chunks are incomplete or do not tile the columns."""
    folder = Path(folder)
    chunk_dir = folder / "chunks" if (folder / "chunks").is_dir() else folder
    found = sorted((int(m.group(1)), int(m.group(2)), p) for p in chunk_dir.glob("chunk_*.npz") if (m := CHUNK.search(p.name)))
    if not found:
        raise FileNotFoundError(f"no chunks in {chunk_dir}")
    totals = {total for _, total, _ in found}
    if len(totals) != 1 or [i for i, _, _ in found] != list(range(totals.pop())):
        raise ValueError(f"{chunk_dir}: chunks incomplete or from different splits")
    loaded = [np.load(p) for _, _, p in found]
    if n_columns is None:
        metas = [json.loads(p.with_suffix(".json").read_text(encoding="utf-8")) for _, _, p in found]
        n_columns = int(metas[0].get("n_columns") or metas[-1]["voxel_stop"])
    index = np.concatenate([c["voxel_index"] for c in loaded])
    out = {}
    for name in metrics:
        if name not in loaded[0]:
            continue
        full = np.full(n_columns, np.nan, dtype=np.float32)
        full[index] = np.concatenate([c[name] for c in loaded])
        out[name] = full
    return out


def stitch(dataset: Dataset, folder: Path, subject: str) -> dict[str, np.ndarray]:
    """Write a fit's chunks as native maps (``<folder>/<metric>.nii.gz``); returns the column vectors."""
    maps = read_chunks(folder, dataset.n_columns(subject))
    for name, values in maps.items():
        dataset.save_map(subject, values, Path(folder) / f"{name}.nii.gz")
    (Path(folder) / "summary.json").write_text(json.dumps(
        {"subject": subject, **{f"mean_{k}": float(np.nanmean(v)) for k, v in maps.items()}}, indent=1), encoding="utf-8")
    return maps
