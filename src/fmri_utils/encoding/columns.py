"""Voxelwise encoding fits on column data (a ``RunTable``), with a fixed test run, in chunks.

``fit_voxelwise`` fits per-voxel ridge through ``fit_encoding`` with a fixed test run: alpha is
chosen per voxel by cross-validation over whole training runs (run i in fold i mod 5, Fisher-z
averaged r), then the model is refitted on every training run and scored by Pearson r on the test
run.

``fit_feature_space`` fits one subject's voxel chunk on a feature space: per-run feature files
``<feature_root>/<feature>/<run>.npy`` (rows = response rows), optionally an extra column appended
after the PCA (``<extra_root>/<run>.npy``, first column), the design of ``design.prepare_design``,
and the noise ceiling from the test run's repeats if the table has them. A chunk is a contiguous
slab of columns: every voxelwise quantity depends only on that voxel, so chunks stitch exactly into
the whole fit. Each chunk writes ``<output>/<model>/<subject>/chunks/chunk_<c>_of_<n>.{npz,json}``
with ``voxel_index``, ``correlation_raw`` and ``selected_alpha`` (and, with repeats,
``noise_ceiling`` and ``correlation_noise_ceiling_normalized``).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Sequence

import numpy as np

from .design import Design, prepare_design
from .noise_ceiling import noise_ceiling, normalise
from .run_table import RunTable, common_finite

#: 10 ... 1e5 in half-decade steps: suits z-scored, delayed designs of a few hundred columns.
WIDE_ALPHAS = tuple(float(10 ** (1 + 0.5 * k)) for k in range(9))
CHUNK = re.compile(r"chunk_(\d{3})_of_(\d{3})\.npz$")


def fit_voxelwise(design: Design, run_ids: Sequence[str], train_responses: Sequence[np.ndarray], test_response: np.ndarray,
                  test_id: str = "test", subject: str = "subject", alphas=WIDE_ALPHAS, inner_splits: int = 5):
    """Per-voxel ridge, alpha by CV over whole training runs; returns (test r, selected alpha) per column."""
    from .config import EncodingConfig
    from .data import EncodingRun, SubjectData
    from .model import fit_encoding
    from .splits import fixed_test_plan
    runs, start = [], 0
    for run_id, response in zip(run_ids, train_responses):
        stop = start + response.shape[0]
        runs.append(EncodingRun(run_id=run_id, features=design.train[start:stop], bold=response,
                                nuisance=np.empty((response.shape[0], 0), dtype=np.float32),
                                source_rows=np.arange(response.shape[0], dtype=np.int32)))
        start = stop
    runs.append(EncodingRun(run_id=test_id, features=design.test, bold=test_response,
                            nuisance=np.empty((test_response.shape[0], 0), dtype=np.float32),
                            source_rows=np.arange(test_response.shape[0], dtype=np.int32)))
    data = SubjectData(subject_id=subject, runs=runs, mask=np.ones((test_response.shape[1], 1, 1), dtype=bool),
                       reference_image=None, voxel_indices=np.arange(test_response.shape[1]))
    config = EncodingConfig(lags=(0,), ridge_alphas=tuple(float(a) for a in alphas), pca_components=(None,),
                            add_run_intercept=False, n_permutations=0)
    result = fit_encoding(data, config, fixed_test_plan(runs, inner_splits))
    return result.mean_correlation.astype(np.float32), result.mean_selected_alpha.astype(np.float32)


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


def load_feature(root: Path, feature: str, run: str) -> np.ndarray:
    path = Path(root) / feature / f"{run}.npy"
    if not path.exists():
        raise FileNotFoundError(path)
    return np.asarray(np.load(path), dtype=np.float32)


class Chunk:
    """One subject's training and test responses for a column chunk, loaded once."""

    def __init__(self, table: RunTable, subject: str, chunk: int = 0, n_chunks: int = 1):
        self.table, self.subject = table, subject
        self.runs = table.runs(subject, "train")
        self.test_run = table.test_run(subject)
        self.n_columns = table.n_columns(subject)
        self.start, self.stop = chunk_bounds(self.n_columns, chunk, n_chunks)
        self.chunk, self.n_chunks = chunk, n_chunks
        train = [table.response(subject, r, self.start, self.stop) for r in self.runs]
        test = table.response(subject, self.test_run, self.start, self.stop)
        repeats = table.repeats(subject, self.test_run, self.start, self.stop)
        if repeats is not None and not np.allclose(repeats.mean(axis=0), test, rtol=1e-5, atol=1e-5, equal_nan=True):
            raise ValueError(f"{subject}: the test response is not the mean of its repeats")
        self.valid = common_finite([*train, test, repeats])
        self.train = [r[:, self.valid] for r in train]
        self.test = test[:, self.valid]
        self.repeats = None if repeats is None else repeats[:, :, self.valid]
        self.voxel_index = (self.start + np.flatnonzero(self.valid)).astype(np.int64)

    def design(self, features: dict[str, np.ndarray], extra_root: Path | None = None, **design_args) -> Design:
        for run, response in zip(self.runs, self.train):
            if features[run].shape[0] != response.shape[0]:
                raise ValueError(f"{self.subject} {run}: features {features[run].shape} vs {response.shape[0]} response rows")
        extra_train = extra_test = None
        if extra_root is not None:
            def extra(run, rows):
                values = np.load(Path(extra_root) / f"{run}.npy").astype(np.float32)
                values = values.reshape(values.shape[0], -1)[:, :1]
                if values.shape[0] != rows:
                    raise ValueError(f"{run}: extra column has {values.shape[0]} rows, not {rows}")
                return values
            extra_train = [extra(r, features[r].shape[0]) for r in self.runs]
            extra_test = extra(self.test_run, features[self.test_run].shape[0])
        return prepare_design([features[r] for r in self.runs], features[self.test_run],
                              extra_train=extra_train, extra_test=extra_test, **design_args)

    def fit(self, design: Design, alphas=WIDE_ALPHAS):
        return fit_voxelwise(design, self.runs, self.train, self.test, self.test_run, self.subject, alphas)


def fit_feature_space(table: RunTable, subject: str, feature_root: Path, feature: str, output: Path, *,
                      model: str | None = None, extra_root: Path | None = None, chunk: int = 0, n_chunks: int = 1,
                      pca_components: int = 256, delays: Sequence[int] = (1, 2, 3, 4), alphas=WIDE_ALPHAS,
                      ceiling_floor: float = 0.3) -> Path:
    """Fit one subject's column chunk on ``<feature_root>/<feature>`` and write the chunk."""
    data = Chunk(table, subject, chunk, n_chunks)
    features = {r: load_feature(feature_root, feature, r) for r in [*data.runs, data.test_run]}
    design = data.design(features, extra_root, pca_components=pca_components, delays=delays)
    correlation, alpha = data.fit(design, alphas)
    arrays = {"correlation_raw": correlation, "selected_alpha": alpha}
    if data.repeats is not None:
        ceiling = noise_ceiling(data.repeats)
        arrays.update(noise_ceiling=ceiling, correlation_noise_ceiling_normalized=normalise(correlation, ceiling, ceiling_floor))
    model = model or feature
    meta = {"subject": subject, "model": model, "feature": feature, "extra_root": str(extra_root) if extra_root else None,
            "voxel_start": data.start, "voxel_stop": data.stop, "n_columns": data.n_columns,
            "n_modeled_voxels": int(data.valid.sum()), "training_runs": data.runs, "test_run": data.test_run,
            "pca_components": design.pca_components, "delays": list(delays), "mean_r": float(np.nanmean(correlation))}
    path = save_chunk(Path(output) / model / subject / "chunks", chunk, n_chunks, data.voxel_index, arrays, meta)
    print(f"[{subject} {model} chunk {chunk}/{n_chunks}] mean r {np.nanmean(correlation):.4f}", flush=True)
    return path


def read_chunks(folder: Path, n_columns: int | None = None) -> dict[str, np.ndarray]:
    """A fit's chunks (``<model>/<subject>``) as column vectors, NaN where unmodelled.

    Raises if the chunks are incomplete or from different splits."""
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
        meta = json.loads(found[-1][2].with_suffix(".json").read_text(encoding="utf-8"))
        n_columns = int(meta.get("n_columns") or meta["voxel_stop"])
    index = np.concatenate([c["voxel_index"] for c in loaded])
    out = {}
    for name in loaded[0].files:
        if name == "voxel_index":
            continue
        full = np.full(n_columns, np.nan, dtype=np.float32)
        full[index] = np.concatenate([c[name] for c in loaded])
        out[name] = full
    return out


def stitch(table: RunTable, folder: Path, subject: str) -> dict[str, np.ndarray]:
    """Write a fit's chunks as maps (``<folder>/<metric>.nii.gz``) through the run table's mask."""
    maps = read_chunks(folder, table.n_columns(subject))
    for name, values in maps.items():
        table.save_map(subject, values, Path(folder) / f"{name}.nii.gz")
    (Path(folder) / "summary.json").write_text(json.dumps(
        {"subject": subject, **{f"mean_{k}": float(np.nanmean(v)) for k, v in maps.items()}}, indent=1), encoding="utf-8")
    return maps
